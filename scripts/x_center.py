import numpy as np

import chainercv

import matplotlib.pyplot as plt

from rsna.datasets.rsna_train_dataset import RSNATrainDataset


def get_center_x(image):
    image = image[0]
    v_sums = np.sum(image, axis=0)
    whole_sum = np.sum(v_sums)
    return np.sum(np.arange(len(v_sums)) * v_sums) / whole_sum


def main():
    dataset = RSNATrainDataset('./RSNA')
    for i in range(100):
        image = dataset[i][1]
        center_x = get_center_x(image)
        bbox = np.array(((0, 0, 1024, center_x),), dtype=np.float32)
        chainercv.visualizations.vis_bbox(np.repeat(image, 3, axis=0), bbox)
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
