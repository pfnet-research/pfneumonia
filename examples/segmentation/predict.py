import argparse
import multiprocessing
import numpy as np
import os
import yaml

import chainer
from chainer.backends import cuda
from chainer.datasets import split_dataset
import chainermn

import matplotlib
matplotlib.use('Agg')

from examples.segmentation.ensemble_model import ModelEnsembler, MultiScaleModelEnsembler
from examples.segmentation.postprocess import Postprocessor, DemoSaver
from examples.segmentation.utils import create_train_val_indices, setup_model

from rsna.datasets.rsna_train_dataset import RSNATrainDataset
from rsna.datasets.rsna_submission_dataset import RSNASubmissionDataset
from rsna.utils.config import load_config
from rsna.utils.predictions import PredictionsManager


def main(commands=None):
    parser = argparse.ArgumentParser(description='Segmentation Predict')
    parser.add_argument('--model', '-m', nargs='+', help='Path to model')
    parser.add_argument('--config', '-c', nargs='*', default=['examples/configs/seg_resnet.yaml'])
    parser.add_argument('--val-set', type=int)
    parser.add_argument('--x-flip', type=int, help='0: no, 1: yes, 2: both (average)', default=0)
    parser.add_argument('--multiscale', action='store_true')

    # Args for ensembling
    parser.add_argument('--ensemble-seg', action='store_true')
    parser.add_argument('--seg-weight', type=float, nargs='*', default=None)
    parser.add_argument('--edge-weight', type=float, nargs='*', default=None)

    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--n-process', '-p', type=int, default=30)
    parser.add_argument('--out', '-o', default='out.csv')

    parser.add_argument('--test', action='store_true')
    parser.add_argument('--limit', '-n', type=int, default=0)
    parser.add_argument('--thresh', '-t', type=float, default=0.1,
                        help='Threshold for edge confidence')

    parser.add_argument('--save-demo-to', metavar='/path/to/out_demo/dir')
    parser.add_argument('--overlay-seg', action='store_true')

    parser.add_argument('--cprofile', action='store_true',
                        help='To profile with cprofile')

    args = parser.parse_args(commands)
    configs = [load_config(yaml.load(open(args.config[i]))) for i in range(len(args.config))]
    master_config = configs[0]

    comm = chainermn.create_communicator(communicator_name='pure_nccl')
    device = comm.intra_rank + args.gpu
    print('Device = {}'.format(device))

    if len(configs) == 1 and len(args.model) >= 2:
        # Duplicate same config
        configs = configs * len(args.model)
    else:
        assert len(configs) == len(args.model), "# of configs and models don't match."

    # Setup models
    models = []
    for i in range(len(args.model)):
        model = setup_model(configs[i], args.x_flip)
        chainer.serializers.load_npz(args.model[i], model)
        models.append(model)

    if len(models) == 1:
        model = models[0]
    else:
        ensembler_cls = MultiScaleModelEnsembler if args.multiscale else ModelEnsembler
        model = ensembler_cls(models, ensemble_seg=args.ensemble_seg,
                              seg_weight=args.seg_weight, edge_weight=args.edge_weight)

    with cuda.get_device_from_id(device):
        model.to_gpu()

    # Setup dataset
    if comm.rank == 0:
        if args.test:
            dataset = RSNASubmissionDataset()
        else:
            if args.val_set is not None:
                master_config['val_set'] = args.val_set
            dataset = RSNATrainDataset()

            if args.val_set is not None:
                master_config['val_set'] = args.val_set

            if master_config['val_set'] == -1:
                val_mask = dataset.patient_df['withinTestRange'].values == 1
                val_indices = val_mask.nonzero()[0]
            else:
                _, val_indices = create_train_val_indices(np.ones(len(dataset), dtype=bool),
                                                          master_config['val_set'])

            dataset = dataset.slice[val_indices, ('dicom_data', 'img', 'bbox')]

        if args.limit and args.limit < len(dataset):
            dataset, _ = split_dataset(dataset, args.limit)
    else:
        dataset = None

    dataset = chainermn.scatter_dataset(dataset, comm)

    if args.cprofile:
        import cProfile
        import pstats
        import io
        pr = cProfile.Profile()
        pr.enable()

    if comm.rank == 0:
        print('Extracting network outputs...')
    outputs = []
    gt_bboxes = []
    for i in range(len(dataset)):
        if comm.rank == 0 and i % 100 == 0:
            print('Processing {}-th sample...'.format(i))
        if args.test:
            dicom_data, image = dataset[i]
            patient_id = dicom_data.PatientID
            gt_bbox = np.empty((0, 4), dtype=np.float32)
        else:
            dicom_data, image, gt_bbox = dataset[i]
            patient_id = dicom_data.PatientID

        if master_config['data_augmentation']['window_width'] > 1.0:
            image = (image - 128) * master_config['data_augmentation']['window_width'] + 128
            image = np.clip(image, 0, 255)

        with cuda.get_device_from_id(device):
            h_seg, h_hor, h_ver = [x[0] for x in model.extract([image])]

        outputs.append((patient_id, image, h_seg, h_hor, h_ver))
        gt_bboxes.append((patient_id, gt_bbox))

    if comm.rank == 0:
        for i in range(1, comm.size):
            other_outputs = comm.recv_obj(i)
            outputs.extend(other_outputs)
            other_gt_bboxes = comm.recv_obj(i)
            gt_bboxes.extend(other_gt_bboxes)
    else:
        comm.send_obj(outputs, 0)
        comm.send_obj(gt_bboxes, 0)
        print('Bye {}.'.format(comm.rank))
        exit(0)

    outputs = sorted(outputs, key=lambda x: x[0])
    gt_bboxes = sorted(gt_bboxes, key=lambda x: x[0])

    print('Done.')
    print('Postprocessing...')
    postprocessor = Postprocessor(master_config['downscale'], args.thresh,
                                  master_config['size_thresh'],
                                  master_config['edge_conf_operation'])
    with multiprocessing.Pool(args.n_process) as p:
        results = p.map(postprocessor.postprocess, outputs)

    results = sorted(results, key=lambda x: x[0])
    print('Done.')

    outputs_ids = [x[0] for x in outputs]
    results_ids = [x[0] for x in results]
    assert outputs_ids == results_ids

    print('Dumping final results...')
    pred_manager = PredictionsManager()
    n_positive = 0
    for result in results:
        patient_id, bbox, label, score = result
        pred_manager.add_prediction(patient_id, bbox, score)
        if len(bbox) > 0:
            n_positive += 1

    print('Complete!')
    print('{} / {} are predicted as positive.'.format(n_positive, len(dataset)))
    with open(args.out, 'w') as f:
        pred_manager.dump(f)

    if args.save_demo_to:
        print('Start saving demos...')
        os.makedirs(args.save_demo_to, exist_ok=True)
        demo_saver = DemoSaver(args.save_demo_to, master_config['downscale'], args.overlay_seg)
        with multiprocessing.Pool(args.n_process) as p:
            p.map(demo_saver.save, list(zip(results, outputs, gt_bboxes)))

    if args.cprofile:
        pr.disable()
        s = io.StringIO()
        sortby = 'time'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()

        print(s.getvalue())
        pr.dump_stats('prof.cprofile'.format(args.out, 0))


if __name__ == '__main__':
    main()
