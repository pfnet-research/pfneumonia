import copy
import git
import os
import time
import yaml

import rsna

default_config = {
    'model_name': 'vgg16',
    'lr': 1e-3,
    'n_gpu': 1,
    'epoch': 20,
    'base_dir': rsna.DEFAULT_BASE_DIR,
    'lr_schedule_args': {},
    'pretrained_model': 'imagenet',
    'roi_pooling_func': 'roi_pooling',  # 'roi_pooling', 'roi_align'
    'use_fpn': False,
    'suppression_method': 'nms',  # 'nms', 'nmw', 'fast_nmw'
    'conf_loss_func': 'softmax',
    'conf_initial_bias': 0,
    'use_psp': False,
    'use_bn_in_fpn_and_head': False,
    'distribute_double': False,
    'n_additional_block': 0,
    'scale': 1,
    'anchor_ratios': [0.5, 1, 2],
    'head_class_name': 'rsna.links.model.fpn.head.Head',
    'head_config': {},
    'batchnorm': 'normal',  # 'fixed', 'normal', 'mn', 'mn_pure_nccl'
    # 'world', 'intra', or positive integer (size of sub groups)
    'batchnorm_comm': -1,
    'min_sizes': None,
    'iterator': 'serial',  # 'serial', 'multithread', 'multiprocess'
    'over_sampling': 1,
    'weight_decay': 0.0001,
}


def complement_config(input_cfg, default_config=default_config,
                      dump_yaml_dir=None):
    output_cfg = copy.copy(default_config)
    for key, val in input_cfg.items():
        if key not in output_cfg:
            raise ValueError('Unknown configuration key: {}'.format(key))
        output_cfg[key] = val
    if dump_yaml_dir is not None:
        os.makedirs(dump_yaml_dir, exist_ok=True)
        cur_time = time.strftime("%Y-%m-%d--%H-%M-%S", time.gmtime())
        dump_yaml_path = os.path.join(
            dump_yaml_dir, '{}.yaml'.format(cur_time))
        with open(dump_yaml_path, 'w') as f:
            yaml.dump(output_cfg, f)
    return output_cfg


def load_git_info():
    repo = git.Repo('.')
    branch_name = repo.active_branch.name
    hexsha = repo.head.commit.hexsha
    commit_message = repo.head.commit.message

    return {'branch': branch_name, 'hexsha': hexsha, 'msg': commit_message}


def load_config(input_cfg, dump_yaml_dir=None):
    """Load config file w/o complementation of default config or checks for keys."""
    output_cfg = {}
    for key, val in input_cfg.items():
        output_cfg[key] = val
    try:
        output_cfg['git'] = load_git_info()
    except:
        print('WARNING: unable to load git info in the current directory.')
    if dump_yaml_dir is not None:
        os.makedirs(dump_yaml_dir, exist_ok=True)
        cur_time = time.strftime("%Y-%m-%d--%H-%M-%S", time.gmtime())
        dump_yaml_path = os.path.join(
            dump_yaml_dir, '{}.yaml'.format(cur_time))
        with open(dump_yaml_path, 'w') as f:
            yaml.dump(dict(output_cfg), f)
    return output_cfg
