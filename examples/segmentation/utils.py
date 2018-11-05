import numpy as np

import chainer
import chainer.links as L
import chainermn

from examples.segmentation.model import UNet, Model, XFlippedModel, XFlippedAndNonFlippedModel

from rsna.extensions import LearningRateScheduler
from rsna.utils.lr_schedules import StepLRSchedule, CosineLRSchedule, CyclicCosineLRSchedule


def create_train_val_indices(mask, val_set):
    """Create indices for train and validation sets.

    It simply splits the dataset evenly into 10 subsets and return `val_set`-th subset (0-origin).
    """
    assert 0 <= val_set <= 9

    n_data = len(mask)
    val_begin = (n_data * val_set) // 10
    val_end = (n_data * (val_set + 1)) // 10

    a = np.arange(n_data)
    val_mask = (val_begin <= a) & (a < val_end)
    train_mask = mask & ~val_mask
    val_mask = mask & val_mask
    return train_mask.nonzero()[0], val_mask.nonzero()[0]


def setup_optimizer(config, comm, model):
    if config['optimizer'] == 'Adam':
        optimizer = chainer.optimizers.Adam(alpha=config['lr'],
                                            weight_decay_rate=config['weight_decay'])
    elif config['optimizer'] == 'MomentumSGD':
        optimizer = chainer.optimizers.MomentumSGD(lr=config['lr'], momentum=config['momentum'])
    else:
        raise ValueError('unknown optimizer type: {}'.format(config['optimizer']))

    optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)
    optimizer.setup(model)

    if config['optimizer'] != 'Adam':
        if config['weight_decay'] > 0:
            optimizer.add_hook(chainer.optimizer.WeightDecay(config['weight_decay']))
    return optimizer


def setup_lr_scheduler(config):
    lr_schedule_args = {
        'base_lr': config['lr'],
    }
    if config.get('resume'):
        config['lr_schedule_args']['warmup_iteration'] = 0
    lr_schedule_args.update(config['lr_schedule_args'])

    if config['lr_schedule'] == 'step':
        lr_schedule = StepLRSchedule(**lr_schedule_args)
    elif config['lr_schedule'] == 'cosine':
        lr_schedule = CosineLRSchedule(**lr_schedule_args)
    elif config['lr_schedule'] == 'cyclic_cosine':
        # Restart every epoch
        lr_schedule = CyclicCosineLRSchedule(n_cycles=config['epoch'], **lr_schedule_args)
    else:
        raise ValueError('Unknown LR Schedule: {}'.format(config['lr_schedule']))

    lr_attr = 'lr' if config['optimizer'] != 'Adam' else 'alpha'
    return LearningRateScheduler(lr_schedule, attr=lr_attr)


def setup_extractor(extractor_name):
    if extractor_name == 'resnet50':
        extractor = L.ResNet50Layers()
    elif extractor_name == 'resnet101':
        extractor = L.ResNet101Layers()
    elif extractor_name == 'resnet152':
        extractor = L.ResNet152Layers()
    else:
        raise ValueError('Unknown extractor name: {}'.format(extractor_name))

    return extractor


def setup_model(config, x_flip):
    extractor = setup_extractor(config['extractor'])
    unet = UNet(extractor, config['use_psp'], config['use_senet'], config['use_coord_conv'])

    model_cls = [Model, XFlippedModel, XFlippedAndNonFlippedModel][x_flip]
    model = model_cls(unet, config['downscale'], config['concat_seg'], config['use_hist_eq'])
    return model
