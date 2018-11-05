import numpy as np

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L


class PyramidPoolingModule(chainer.ChainList):

    def __init__(self, out_channels_per_pyramid, pyramid_sizes):
        super().__init__()
        self.pyramid_sizes = pyramid_sizes
        with self.init_scope():
            for _ in pyramid_sizes:
                self.add_link(L.Convolution2D(None, out_channels_per_pyramid, 1, 1, 0))

    def __call__(self, x):
        ys = [x]
        H, W = x.shape[2:]
        for f, pyramid_size in zip(self.children(), self.pyramid_sizes):
            assert H % pyramid_size == 0 and W % pyramid_size == 0,\
                'Height and width must be divisors of the pyramid size in PSP layer.'
            y = F.average_pooling_2d(x, H // pyramid_size, W // pyramid_size)
            y = f(y)  # Reduce num of channels
            y = F.resize_images(y, (H, W))
            ys.append(y)
        return F.concat(ys, axis=1)


def make_coord_conv(xp, shape):
    N, _, H, W = shape
    y_coord = xp.linspace(-1, 1, H, dtype=np.float32)
    x_coord = xp.linspace(-1, 1, W, dtype=np.float32)
    y_mesh, x_mesh = xp.meshgrid(y_coord, x_coord, indexing='ij')
    to_shape = (N, 1, H, W)
    return xp.broadcast_to(y_mesh, to_shape), xp.broadcast_to(x_mesh, to_shape)


class UNet(chainer.Chain):

    def __init__(self, extractor, use_psp, use_senet, use_coord_conv,
                 layers=('res5', 'res4', 'res3', 'res2', 'conv1'),
                 n_channels=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.use_psp = use_psp
        self.use_senet = use_senet
        self.use_coord_conv = use_coord_conv
        self.layers = layers
        self.n_channels = n_channels
        resnet_layers = (2048, 1024, 512, 256, 64)

        with self.init_scope():
            self.extractor = extractor
            for i in range(len(layers)):
                layer = layers[i]
                n_channel = n_channels[i]
                if use_senet:
                    out_size = resnet_layers[i]
                    if i > 0:
                        out_size += n_channels[i - 1]
                    senet = SENet(axis=1, out_size=out_size, rate=16)
                    setattr(self, 'senet_{}'.format(layer), senet)
                conv1 = L.Convolution2D(None, n_channel, 3, 1, 1)
                setattr(self, 'conv1_{}'.format(layer), conv1)
                conv2 = L.Convolution2D(None, n_channel, 3, 1, 1)
                setattr(self, 'conv2_{}'.format(layer), conv2)

            if use_psp:
                self.psp = PyramidPoolingModule(512, (1, 2, 4, 8))

    def __call__(self, images):
        feature_maps = self.extractor.__call__(images, layers=self.layers)
        h = feature_maps[self.layers[0]]

        for i, layer in enumerate(self.layers):
            if i == 0 and self.use_psp:
                h = self.psp(h)
            elif i > 0:
                h = F.unpooling_2d(h, 2, outsize=(h.shape[2] * 2, h.shape[3] * 2))
                h = F.concat((feature_maps[layer], h))

            if self.use_coord_conv:
                y_mesh, x_mesh = make_coord_conv(self.xp, h.shape)
                h = F.concat((h, y_mesh, x_mesh))

            if self.use_senet:
                senet = getattr(self, 'senet_{}'.format(layer))
                h = senet(h)

            conv1 = getattr(self, 'conv1_{}'.format(layer))
            h = F.relu(conv1(h))
            conv2 = getattr(self, 'conv2_{}'.format(layer))
            h = F.relu(conv2(h))

        return h


class SENet(chainer.Chain):

    def __init__(self, axis, out_size, rate):
        super().__init__()
        self.axis = axis
        self.out_size = out_size
        self.rate = rate

        with self.init_scope():
            self.fc1 = L.Linear(out_size // rate)
            self.fc2 = L.Linear(out_size)

    def __call__(self, x):
        h = F.average(x, axis=tuple([i for i in range(1, 4) if i != self.axis]))
        h = F.relu(self.fc1(h))
        h = F.sigmoid(self.fc2(h))
        shape = [x.shape[0], 1, 1, 1]
        shape[self.axis] = self.out_size
        h = x * F.broadcast_to(h.reshape(shape), x.shape)
        return h


class Model(chainer.Chain):

    def __init__(self, unet, downscale, concat_seg, use_hist_eq, postprocessor=None):
        super().__init__()
        self.downscale = downscale
        self.concat_seg = concat_seg
        self.use_hist_eq = use_hist_eq
        self.postprocessor = postprocessor

        with self.init_scope():
            self.unet = unet

            self.seg = L.Convolution2D(None, 1, 1, 1, 0)
            self.last_conv1 = L.Convolution2D(None, 64, 3, 1, 1)
            self.last_conv2 = L.Convolution2D(None, 64, 3, 1, 1)
            self.hor = L.Convolution2D(None, 2, 1, 1, 0)
            self.ver = L.Convolution2D(None, 2, 1, 1, 0)

    def __call__(self, images):
        h = self.forward_backbone(images)
        h_seg = self.forward_seg(h)
        h_hor, h_ver = self.forward_edge(h, h_seg)
        return h_seg, h_hor, h_ver

    def forward_backbone(self, images):
        return self.unet(images)

    def forward_seg(self, h):
        h_seg = self.seg(h)  # (N, 1, H, W)
        return h_seg

    def forward_edge(self, h, h_seg):
        if self.concat_seg == 2:
            h = F.concat((h, h_seg.array, F.sigmoid(h_seg.array)))
        elif self.concat_seg == 1:
            h = F.concat((h, F.sigmoid(h_seg.array)))
        h = F.relu(self.last_conv1(h))
        h = F.relu(self.last_conv2(h))
        h_hor = self.hor(h)  # (N, 2, H, W)
        h_ver = self.ver(h)  # (N, 2, H, W)
        return h_hor, h_ver

    def prepare(self, images, downscale=1):
        if self.use_hist_eq:
            # Histogram equalization
            images = images.astype(int, copy=False)
            values, counts = self.xp.unique(images, return_counts=True)
            p = self.xp.zeros(256, dtype=int)
            p[values] = counts
            d1 = 255 / images.size * p.cumsum().astype(np.float32, copy=False)
            images = d1[images]

        images = F.repeat(images, 3, axis=1)
        if downscale > 1:
            output_shape = [x // downscale for x in images.shape[2:]]
            images = F.resize_images(images, tuple(output_shape))
        images -= self.xp.array([103.063, 115.903, 123.152], dtype=np.float32).reshape(1, 3, 1, 1)
        return images

    def extract(self, images):
        prepared_images = self.xp.asarray(images)
        prepared_images = self.prepare(prepared_images, self.downscale)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            h_seg, h_hor, h_ver = self.__call__(prepared_images)
        h_seg = cuda.to_cpu(h_seg.array)
        h_hor = cuda.to_cpu(h_hor.array)
        h_ver = cuda.to_cpu(h_ver.array)
        return h_seg, h_hor, h_ver

    def predict(self, images, return_raw_data=False):
        h_seg, h_hor, h_ver = self.extract(images)
        bboxes = []
        labels = []
        scores = []
        for i in range(len(images)):
            bbox, label, score = self.postprocessor(images[i], h_seg[i], h_hor[i], h_ver[i])
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)
        if return_raw_data:
            return bboxes, labels, scores, h_seg, h_hor, h_ver
        else:
            return bboxes, labels, scores


class XFlippedModel(Model):

    def forward_backbone(self, images):
        h = super().forward_backbone(images[:, :, :, ::-1])
        return h

    def forward_seg(self, h):
        h_seg = super().forward_seg(h)
        return h_seg[:, :, :, ::-1]

    def forward_edge(self, h, h_seg):
        h_hor, h_ver = super().forward_edge(h, h_seg[:, :, :, ::-1])
        h_hor = h_hor[:, :, :, ::-1]
        h_ver = h_ver[:, ::-1, :, ::-1]  # left edge becomes right, vice versa
        return h_hor, h_ver


class XFlippedAndNonFlippedModel(Model):

    def forward_backbone(self, images):
        h1 = super().forward_backbone(images)
        h2 = super().forward_backbone(images[:, :, :, ::-1])
        return h1, h2

    def forward_seg(self, h):
        h1, h2 = h
        h_seg1 = super().forward_seg(h1)
        h_seg2 = super().forward_seg(h2)[:, :, :, ::-1]
        return (h_seg1 + h_seg2) / 2

    def forward_edge(self, h, h_seg):
        h1, h2 = h
        h_hor1, h_ver1 = super().forward_edge(h1, h_seg)
        h_hor2, h_ver2 = super().forward_edge(h2, h_seg[:, :, :, ::-1])
        h_hor2 = h_hor2[:, :, :, ::-1]
        h_ver2 = h_ver2[:, ::-1, :, ::-1]  # left edge becomes right, vice versa
        return (h_hor1 + h_hor2) / 2, (h_ver1 + h_ver2) / 2
