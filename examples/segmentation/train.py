import argparse
from PIL import Image
import numpy as np
import sys
import yaml

import chainer
from chainer.backends import cuda
from chainer.dataset import DatasetMixin
from chainer.datasets import TransformDataset
from chainer.iterators import MultiprocessIterator
import chainer.functions as F
from chainer.training import extensions
from chainer.training.triggers import ManualScheduleTrigger
from chainercv import transforms
import chainermn

sys.path.append('.')

from examples.segmentation.utils import create_train_val_indices, setup_optimizer,\
    setup_lr_scheduler, setup_model
import examples.segmentation.predict as predict

from rsna.datasets.rsna_train_dataset import RSNATrainDataset
from rsna.utils.config import load_config
from rsna.utils.dataset.dataset import oversample_dataset
from rsna.utils.transforms import rotate_bbox


def compute_l4_mean(x):
    x = x.astype(np.float64)
    n = x.size
    x_sum = x.sum()
    x2 = x * x
    x2_sum = x2.sum()
    x3 = x2 * x
    x3_sum = x3.sum()

    def grad(t):
        return n * t * t * t - 3 * x_sum * t * t + 3 * x2_sum * t - x3_sum

    left, right = 0, 255
    for i in range(10):
        mid = (left + right) / 2
        if grad(mid) < 0:
            left = mid
        else:
            right = mid
    return (left + right) / 2


def get_area_sum(cumsum, y1, x1, y2, x2):
    if y1 < 0 or x1 < 0 or y2 >= cumsum.shape[0] or x2 >= cumsum.shape[1]:
        return 0
    return cumsum[y2, x2] - cumsum[y2, x1] - cumsum[y1, x2] + cumsum[y1, x1]


def compute_f1(a, b):
    tp = F.sum(F.minimum(a, b))
    error = F.sum(F.absolute(a - b))
    f1 = 2 * tp / (2 * tp + error)
    return f1


def compute_accuracy(a, b):
    return a * b + (1 - a) * (1 - b)


def logit(a, xp):
    return xp.log(a / (1 - a))


def compute_logit_loss(a, b):
    return F.squared_error(logit(a, F), logit(b, F))


def soft_label_cross_entropy(y, t):
    # y and t must be within [0, 1].
    eps = 1e-7
    t_entropy = - (t * F.log(t + eps) + (1 - t) * F.log(1 - t + eps))
    return - (t * F.log(y + eps) + (1 - t) * F.log(1 - y + eps)) - t_entropy


class TrainChain(chainer.Chain):

    def __init__(self, model, downscale):
        super().__init__()
        self.downscale = downscale
        with self.init_scope():
            self.model = model

    def __call__(self, images, labels, bboxes, masks):
        """

        Args:
            images: (N, 1, 1024, 1024)
            labels: (N,)
            bboxes: (N, R, 4), where `R` is the maximum number of bbox over the batch.
            masks: (N, 1, 1024, 1024)

        Returns:

        """
        images = self.model.prepare(images, self.downscale)
        H, W = images.shape[2:]
        masks = F.average_pooling_2d(masks, self.downscale * 2).array

        h_seg, h_hor, h_ver = self.model(images)

        pred_seg = F.sigmoid(h_seg)
        seg_f1 = compute_f1(pred_seg, masks)
        seg_loss = 1 - seg_f1

        seg_acc = F.average(compute_accuracy(pred_seg, masks), axis=(1, 2, 3)).array

        raw_bboxes = cuda.to_cpu(bboxes)
        N, R, _ = raw_bboxes.shape
        raw_bboxes[:, :, 2:] -= 1  # to closed interval

        gt_hor_edge = self.xp.zeros((N, 2, H // 2, W // 2), dtype=np.int32)
        gt_ver_edge = self.xp.zeros((N, 2, H // 2, W // 2), dtype=np.int32)
        for b in range(N):
            for r in range(R):
                if raw_bboxes[b, r, 0] == 0:
                    break
                pos = raw_bboxes[b, r, :].astype(int) // (self.downscale * 2)
                gt_hor_edge[b, 0, pos[0], pos[1]:pos[3] + 1] = 1
                gt_hor_edge[b, 1, pos[2], pos[1]:pos[3] + 1] = 1
                gt_ver_edge[b, 0, pos[0]:pos[2] + 1, pos[1]] = 1
                gt_ver_edge[b, 1, pos[0]:pos[2] + 1, pos[3]] = 1

        # loss for edges
        # undersample negative cells
        undersampling_rate = 1 / (H // 2 + W // 2)
        gt_hor_edge -= (self.xp.random.rand(N, 2, H // 2, W // 2) > undersampling_rate) * (1 - gt_hor_edge)
        gt_ver_edge -= (self.xp.random.rand(N, 2, H // 2, W // 2) > undersampling_rate) * (1 - gt_ver_edge)
        edge_loss = []
        for i in range(N):
            hor_loss = F.sigmoid_cross_entropy(h_hor[i:i + 1], gt_hor_edge[i:i + 1])
            ver_loss = F.sigmoid_cross_entropy(h_ver[i:i + 1], gt_ver_edge[i:i + 1])
            edge_loss.append(hor_loss + ver_loss)
        edge_loss = F.stack(edge_loss)
        raw_edge_loss = F.average(edge_loss)
        edge_loss = F.average(edge_loss * seg_acc * seg_acc)

        loss = seg_loss + edge_loss

        chainer.reporter.report({'loss': loss,
                                 'seg_f1': seg_f1,
                                 'seg_loss': seg_loss,
                                 'edge_loss': edge_loss,
                                 'raw_edge_loss': raw_edge_loss,
                                 }, self)
        return loss


N = 0
def save_data(image, bbox, mask):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from chainercv import visualizations

    img = np.concatenate((image, image, mask * 255), axis=0)
    visualizations.vis_bbox(img, bbox)
    global N
    plt.savefig('data/{}.png'.format(N))
    N += 1
    plt.close()


class Augment(object):

    def __init__(self, downscale, options):
        self.downscale = downscale
        self.options = options.copy()

    def __call__(self, in_data):
        image, label, bbox, mask = in_data
        _, H, W = image.shape
        cell_size = 32 * self.downscale

        # Horizontal flip
        if self.options['x_flip']:
            if np.random.randint(2) == 0:
                image = transforms.flip(image, x_flip=True)
                bbox = transforms.flip_bbox(bbox, (H, W), x_flip=True)
                mask = transforms.flip(mask, x_flip=True)

        # Random rotation (90 or 270 degrees)
        if self.options['rotate90']:
            assert H == W, 'Height and width must match when `rotate90` is set.'
            if np.random.randint(2) == 0:  # Rotate?
                if np.random.randint(2) == 0:  # Counter-clockwise?
                    bbox = rotate_bbox(bbox, 1, (H, W))
                    image = np.rot90(image, 1, axes=(1, 2))
                    mask = np.rot90(mask, 1, axes=(1, 2))
                else:
                    bbox = rotate_bbox(bbox, 3, (H, W))
                    image = np.rot90(image, 3, axes=(1, 2))
                    mask = np.rot90(mask, 3, axes=(1, 2))
                _, H, W = image.shape

        # Zoom in / zoom out
        if self.options['zoom'] > 1:
            assert self.options['scale'] <= 1.0, "`scale` shouldn't be set if `zoom` is set."
            max_log_zoom = np.log(self.options['zoom'])
            log_zoom = np.random.random() * 2 * max_log_zoom - max_log_zoom
            zoom = np.exp(log_zoom)

            if zoom > 1:
                # Zoom in
                y_size, x_size = int(H / zoom), int(W / zoom)
                y_offset = np.random.randint(H - y_size + 1)
                x_offset = np.random.randint(W - x_size + 1)
                y_slice = slice(y_offset, y_offset + y_size)
                x_slice = slice(x_offset, x_offset + x_size)
                bbox = transforms.crop_bbox(bbox, y_slice, x_slice)

                bbox *= zoom
                image = transforms.resize(image[:, y_slice, x_slice], (H, W))
                mask = transforms.resize(mask[:, y_slice, x_slice], (H, W), interpolation=Image.NEAREST)
            elif zoom < 1:
                # Zoom out
                y_size, x_size = int(H / zoom), int(W / zoom)
                y_offset = np.random.randint(y_size - H + 1)
                x_offset = np.random.randint(x_size - W + 1)

                bbox = transforms.translate_bbox(bbox, y_offset, x_offset)
                new_image = np.zeros((1, y_size, x_size), dtype=np.float32)
                new_image[:, y_offset:y_offset + H, x_offset:x_offset + W] = image
                new_mask = np.zeros((1, y_size, x_size), dtype=np.float32)
                new_mask[:, y_offset:y_offset + H, x_offset:x_offset + W] = mask

                bbox *= zoom
                image = transforms.resize(new_image, (H, W))
                mask = transforms.resize(new_mask, (H, W), interpolation=Image.NEAREST)

        # Random scale
        if self.options['scale'] > 1.0:
            assert self.options['crop'], '`crop` must be set if `scale` is set.'
            max_log_scale = np.log(self.options['scale'])
            log_scale = np.random.random() * 2 * max_log_scale - max_log_scale
            scale = np.exp(log_scale)

            image = transforms.resize(image, (int(H * scale), int(W * scale)))
            mask = transforms.resize(mask, (int(H * scale), int(W * scale)), interpolation=Image.NEAREST)
            _, H, W = image.shape
            bbox *= scale

        # Random crop
        if self.options['crop']:
            y_margin = (H - 1) % cell_size + 1
            x_margin = (W - 1) % cell_size + 1
            y_offset = np.random.randint(y_margin)
            x_offset = np.random.randint(x_margin)
            y_size = H - y_margin
            x_size = W - x_margin
            y_slice = slice(y_offset, y_offset + y_size)
            x_slice = slice(x_offset, x_offset + x_size)

            image = image[:, y_slice, x_slice]
            bbox = transforms.crop_bbox(bbox, y_slice, x_slice)
            mask = mask[:, y_slice, x_slice]

        # Change window width
        if self.options['window_width'] > 1.0:
            image = (image - 128) * self.options['window_width'] + 128

        # Change contrast
        if self.options['contrast']:
            image += np.random.randint(self.options['contrast'] * 2 + 1) - self.options['contrast']

        image = np.clip(image, 0, 255)
        # save_data(image, bbox, mask)
        return image, label, bbox, mask


def hist_eq(x):
    p = np.empty(256)
    for i in range(256):
        p[i] = ((x == i).sum())
    d1 = 255 / x.size * p.cumsum().astype(np.float32)
    return d1[x.astype(int)]


def preprocess(in_data):
    _, image, label, bbox, _, _ = in_data
    mask = np.zeros(image.shape, dtype=np.float32)
    for i in range(len(bbox)):
        y_min, x_min, y_max, x_max = bbox[i].astype(int)
        mask[:, y_min:y_max, x_min:x_max] = 1

    return image, label, bbox, mask


class CachedDataset(DatasetMixin):
    def __init__(self, base):
        self.base = base
        self.cache = {}

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        if i not in self.cache:
            self.cache[i] = self.base.get_example(i)
        return self.cache[i]


def main():
    parser = argparse.ArgumentParser(description='Segmentation model')
    parser.add_argument('--config', '-c', default='examples/configs/seg_resnet.yaml')
    parser.add_argument('--out', '-o', default='results',
                        help='Output directory')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--snapshot', type=int, help='Snapshot interval', default=1)
    parser.add_argument('--val-set', type=int)
    parser.add_argument('--predict', action='store_true')

    parser.add_argument('--benchmark', action='store_true',
                        help='To run benchmark mode')
    parser.add_argument('--benchmark-iterations', type=int, default=500,
                        help='the number of iterations when using benchmark mode')
    parser.add_argument('--cprofile', action='store_true',
                        help='To profile with cprofile')

    args = parser.parse_args()
    config = load_config(yaml.load(open(args.config)), dump_yaml_dir=args.out)

    comm = chainermn.create_communicator(communicator_name='pure_nccl')
    device = comm.intra_rank + args.gpu
    cuda.get_device_from_id(device).use()
    if comm.size != config['n_gpu']:
        raise ValueError('# of GPUs specified in config file does not match '
                         'the actual number of available GPUs. '
                         'Expected={} Actual={}'.format(config['n_gpu'], comm.size))

    if args.val_set is not None:
        assert 0 <= args.val_set <= 9
        config['val_set'] = args.val_set

    trainer_stop_trigger = config["epoch"], 'epoch'
    if args.benchmark:
        trainer_stop_trigger = args.benchmark_iterations, 'iteration'

    # Setup model
    model = setup_model(config, 0)
    if config.get('resume'):
        chainer.serializers.load_npz(config['resume'], model)
    train_chain = TrainChain(model, config['downscale'])
    train_chain.to_gpu()

    # Setup dataset
    if comm.rank == 0:
        dataset = RSNATrainDataset()

        # Determine samples to pick up
        assert config['view_position'] in ('both', 'pa', 'ap', 'no-pa-pos')
        if config['view_position'] == 'both':
            mask = np.ones(len(dataset), dtype=bool)
        elif config['view_position'] == 'no-pa-pos':
            mask = dataset.patient_df['ViewPosition'].values == 'PA'
            mask &= dataset.get_mask_for_positive_samples()
            mask = ~mask
        else:
            mask = dataset.patient_df['ViewPosition'].values == 'PA'
            if config['view_position'] == 'ap':
                mask = ~mask

        if config['val_set'] == -1:
            train_mask = mask & (dataset.patient_df['withinTestRange'].values == 0)
            train_indices = train_mask.nonzero()[0]
            val_mask = mask & (dataset.patient_df['withinTestRange'].values == 1)
            val_indices = val_mask.nonzero()[0]
        else:
            train_indices, val_indices = create_train_val_indices(mask, config['val_set'])
        train_data = dataset.slice[train_indices]
        val_data = dataset.slice[val_indices]
        print('train = {}, val = {}'.format(len(train_data), len(val_data)))

        positive_mask = dataset.get_mask_for_positive_samples()[train_indices]
        if config['oversampling_rate'] > 1:
            train_data = oversample_dataset(train_data, positive_mask, config['oversampling_rate'])
            print('==> train = {} ({}x oversampled with {} positive samples)'.format(
                len(train_data), config['oversampling_rate'], positive_mask.sum()))
        else:
            print('--> no oversampling with {} positive samples'.format(positive_mask.sum()))

        train_data = TransformDataset(train_data, preprocess)
        val_data = TransformDataset(val_data, preprocess)

        # Data augmentation
        augment = Augment(config['downscale'], config['data_augmentation'])
        train_data = TransformDataset(train_data, augment)
    else:
        train_data, val_data = None, None

    train_data = chainermn.scatter_dataset(train_data, comm)
    val_data = chainermn.scatter_dataset(val_data, comm)

    # Setup iterator, optimizer and updater
    train_iter = MultiprocessIterator(train_data, batch_size=config['batch_size'],
                                      shared_mem=10000000)
    val_iter = MultiprocessIterator(val_data, batch_size=config['batch_size'],
                                    repeat=False, shuffle=False,
                                    shared_mem=10000000)

    optimizer = setup_optimizer(config, comm, train_chain)
    if not config.get('resume') and config['extractor_freeze_iteration'] != 0:
        model.unet.extractor.disable_update()

    updater = chainer.training.updaters.StandardUpdater(
        train_iter, optimizer, device=device,
        converter=lambda x, y: chainer.dataset.concat_examples(x, y, 0))

    # Setup trainer
    trainer = chainer.training.Trainer(updater,
                                       stop_trigger=trainer_stop_trigger,
                                       out=args.out)

    trainer.extend(setup_lr_scheduler(config), trigger=(1, 'iteration'))

    if comm.rank == 0:
        log_interval = 10, 'iteration'
        print_interval = 10, 'iteration'

        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.observe_lr(), trigger=print_interval)
        entries = ['iteration', 'epoch', 'elapsed_time', 'lr']
        measurements = ['loss', 'seg_f1', 'seg_loss', 'edge_loss', 'raw_edge_loss']
        entries.extend(['main/{}'.format(x) for x in measurements])
        entries.extend(['validation/main/{}'.format(x) for x in measurements])
        trainer.extend(extensions.PrintReport(entries), trigger=print_interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))

        if args.snapshot > 0:
            trainer.extend(extensions.snapshot_object(model, 'model_epoch_{.updater.epoch}.npz'),
                           trigger=(args.snapshot, 'epoch'))
        trainer.extend(extensions.snapshot_object(model, 'final_model.npz'),
                       trigger=trainer_stop_trigger)

    evaluator = extensions.Evaluator(
        val_iter, train_chain, device=device,
        converter=lambda x, y: chainer.dataset.concat_examples(x, y, 0))
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator, trigger=(1, 'epoch'))

    @chainer.training.make_extension(trigger=(1, 'epoch'), priority=-100)
    def enable_extractor_update(_):
        print('enable update!')
        model.unet.extractor.enable_update()

    if config['extractor_freeze_iteration'] > 0:  # no melt if -1
        melt_trigger = ManualScheduleTrigger(config['extractor_freeze_iteration'], 'iteration')
        trainer.extend(enable_extractor_update, trigger=melt_trigger)

    trainer.run()

    if args.predict:
        if comm.rank == 0:
            commands = ['--out', '{}/t0.01.csv'.format(args.out),
                        '--model', '{}/final_model.npz'.format(args.out),
                        '--config', args.config,
                        '--val-set', str(config['val_set']),
                        '--gpu', str(args.gpu),
                        '--thresh', '0.01',
                        ]
            predict.main(commands)

            commands[1] = '{}/test-t0.01.csv'.format(args.out)
            commands.append('--test')
            predict.main(commands)


if __name__ == '__main__':
    main()
