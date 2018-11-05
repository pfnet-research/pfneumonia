import numpy as np

import chainer
from chainer.backends import cuda
import chainer.functions as F


class ModelEnsembler(chainer.Chain):
    """Ensemble two or more models.

    Args:
        models (iterable of rsna.segmentation.model.Model): Models.
        ensemble_seg (bool): If `True`, average `h_seg` over models.
        seg_weight (iterable of float): Segmentation weight for each model. It makes sense only
            if `ensemble_seg` is `True`.
        edge_weight (iterable of float): Edge weight for each model. The final values of `h_hor`
            and `h_ver` will be the weighted mean of `h_hor` and `h_ver` of each model with this
            weight.
    """

    def __init__(self, models, ensemble_seg=True, seg_weight=None, edge_weight=None):
        super().__init__()
        self.ensemble_seg = ensemble_seg

        if seg_weight is None:
            seg_weight = np.ones(len(models))
        if edge_weight is None:
            edge_weight = np.ones(len(models))
        self.seg_weight = np.array(seg_weight, dtype=np.float32)
        self.edge_weight = np.array(edge_weight, dtype=np.float32)

        with self.init_scope():
            self.models = chainer.ChainList()
            for model in models:
                self.models.add_link(model)

    def __call__(self, images):
        hs = []
        for model in self.models:
            hs.append(model.forward_backbone(images))

        h_segs = []
        for (h, model) in zip(hs, self.models):
            h_segs.append(model.forward_seg(h))
        h_segs = F.stack(h_segs)
        h_seg_avg = F.average(h_segs, axis=0, weights=self.xp.asarray(self.seg_weight))

        if self.ensemble_seg:
            h_segs = [h_seg_avg] * len(self.models)

        h_hors, h_vers = [], []
        for i in range(len(self.models)):
            h_hor, h_ver = self.models[i].forward_edge(hs[i], h_segs[i])
            h_hors.append(h_hor)
            h_vers.append(h_ver)
        h_hors = F.stack(h_hors)
        h_hor_avg = F.average(h_hors, axis=0, weights=self.xp.asarray(self.edge_weight))
        h_vers = F.stack(h_vers)
        h_ver_avg = F.average(h_vers, axis=0, weights=self.xp.asarray(self.edge_weight))

        return h_seg_avg, h_hor_avg, h_ver_avg

    def extract(self, images):
        prepared_images = self.xp.asarray(images)
        prepared_images = self.models[0].prepare(prepared_images, self.models[0].downscale)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            h_seg, h_hor, h_ver = self.__call__(prepared_images)
        h_seg = cuda.to_cpu(h_seg.array)
        h_hor = cuda.to_cpu(h_hor.array)
        h_ver = cuda.to_cpu(h_ver.array)
        return h_seg, h_hor, h_ver


class MultiScaleModelEnsembler(ModelEnsembler):

    scales = (1, 13 / 16, 19 / 16)  # First element should always be 1
    weights = (0.5, 0.25, 0.25)

    def extract(self, images):
        prepared_images = self.xp.asarray(images)
        prepared_images = self.models[0].prepare(prepared_images, 1)

        h_segs, h_hors, h_vers = [], [], []
        for i in range(len(self.scales)):
            H, W = prepared_images.shape[2:]
            hh = int((H // self.models[0].downscale) * self.scales[i])
            ww = int((W // self.models[0].downscale) * self.scales[i])
            resized_prepared_images = F.resize_images(prepared_images, (hh, ww))

            with chainer.using_config('train', False), chainer.no_backprop_mode():
                h_seg, h_hor, h_ver = self.__call__(resized_prepared_images)

            if self.scales[i] != 1:
                h_seg = F.resize_images(h_seg, (512 // self.models[0].downscale,
                                                512 // self.models[0].downscale))
                h_hor = F.resize_images(h_hor, (512 // self.models[0].downscale,
                                                512 // self.models[0].downscale))
                h_ver = F.resize_images(h_ver, (512 // self.models[0].downscale,
                                                512 // self.models[0].downscale))
            weight = self.weights[i] / sum(self.weights)
            h_segs.append(h_seg.array * weight)
            h_hors.append(h_hor.array * weight)
            h_vers.append(h_ver.array * weight)
        h_seg = cuda.to_cpu(sum(h_segs))
        h_hor = cuda.to_cpu(sum(h_hors))
        h_ver = cuda.to_cpu(sum(h_vers))
        return h_seg, h_hor, h_ver
