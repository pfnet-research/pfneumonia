import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import chainercv

from rsna.datasets.rsna_train_dataset import RSNATrainDataset
from rsna.utils import PredictionsManager


def main():
    parser = argparse.ArgumentParser(
        description='Script to evaluate results of validation data')
    parser.add_argument('--input', '-i', type=str,
                        help='submission file of validation dataset')
    parser.add_argument('--thresh', '-t', type=float, nargs='*', default=[])
    parser.add_argument('--view-position', '--vp', default='both', choices=('both', 'ap', 'pa'))
    parser.add_argument('--stage1-test', action='store_true',
                        help='If true, evaluate only on Stage1 test set.')
    args = parser.parse_args()

    dataset = RSNATrainDataset()

    pred = PredictionsManager()
    pred.restore(args.input)

    if args.stage1_test:
        p = Path('./RSNA/stage_1_test_images')
        print('Searching {}...'.format(p))
        stage1_test_ids = set([x.stem for x in p.glob('*.dcm')])
        assert len(stage1_test_ids) == 1000
        print('Picking up Stage1 test set...')
        pred2 = PredictionsManager()
        for key, item in pred.predictions.items():
            if key in stage1_test_ids:
                pred2.add_prediction(key, item['bbox'], item['score'])
        print('Successfully loaded {} patients.'.format(len(pred2.predictions)))
        pred = pred2

    iou_thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]

    fp_cutoffs = []
    tp_cutoffs = []
    tp_score_gains = []
    n_targets = 0
    for patient_id, entry in pred.predictions.items():
        patient_index = dataset.get_index_with_patient_id(patient_id)
        gt_bbox = dataset.get_bbox(patient_index)
        view_position = dataset.get_view_position(patient_index)
        bbox, score = entry['bbox'], entry['score']

        if ((args.view_position == 'ap' and view_position == 'PA') or
            (args.view_position == 'pa' and view_position == 'AP')):
            continue

        if len(gt_bbox) > 0:
            n_targets += 1

        if len(bbox) == 0:
            # Skip negative predictions
            pass
        elif len(gt_bbox) == 0 and len(bbox) > 0:
            # FP
            fp_cutoffs.append(score.max())
        else:
            # TP
            n_tps = np.zeros(len(iou_thresholds))
            order = score.argsort()[::-1]  # sort by score (decreased order)
            bbox = bbox[order]
            score = score[order]
            bbox_iou = chainercv.utils.bbox_iou(bbox, gt_bbox)
            max_ious = bbox_iou.max(axis=1)
            gt_indices = bbox_iou.argmax(axis=1)
            gt_used = np.zeros((len(iou_thresholds), len(gt_bbox)), dtype=bool)
            prev_ap = 0
            for k, (iou, conf) in enumerate(zip(max_ious, score)):
                for i in range(len(iou_thresholds)):
                    if not gt_used[i, gt_indices[k]] and iou >= iou_thresholds[i]:
                        n_tps[i] += 1
                        gt_used[i, gt_indices[k]] = True
                # TP + FP = (# of predictions), TP + FN = len(gt_bbox)
                # thus TP + FP + FN = (# of predictions) + len(gt_bbox) - TP
                prec = n_tps / ((k + 1) + len(gt_bbox) - n_tps)
                ap = prec.mean()
                tp_cutoffs.append(conf)
                tp_score_gains.append(ap - prev_ap)
                prev_ap = ap

    cutoffs = np.array(tp_cutoffs + fp_cutoffs)
    order = cutoffs.argsort()[::-1]
    score_gains = np.array(tp_score_gains + [-2] * len(fp_cutoffs))  # use '-2' as fp
    cutoffs = cutoffs[order]
    score_gains = score_gains[order]

    current_score = 0  # sum of score for TP images
    best_map = 0
    best_thresh = 1
    n_fp = 0
    maps = []
    current_map = 0
    for cutoff, score_gain in zip(cutoffs, score_gains):
        if score_gain == -2:
            n_fp += 1
        else:
            current_score += score_gain
        current_map = current_score / (n_targets + n_fp)
        maps.append(current_map)
        if current_map > best_map:
            best_map = current_map
            best_thresh = cutoff

    print('Current MAP = {:.5f}'.format(current_map))
    for t in args.thresh:
        index = len(cutoffs) - 1 - np.searchsorted(cutoffs[::-1], t)
        m = 0 if index < -1 else maps[index]
        print('MAP for t={:.2f}: {:.5f}'.format(t, m))
    print('Mean MAP = {:.5f}'.format(sum(maps) / len(maps)))
    print('Best threshold = {:.5f}'.format(best_thresh))
    print('Best MAP = {:.5f}'.format(best_map))

    try:
        plt.plot(cutoffs, maps)
        plt.xlabel('threshold')
        plt.ylabel('mAP')
        plt.show()
    except:  # NOQA
        print('Unable to plot the figure. Try X-forwarding.')


if __name__ == '__main__':
    main()
