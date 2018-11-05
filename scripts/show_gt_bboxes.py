import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from chainercv.visualizations import vis_bbox

from rsna.datasets.rsna_train_dataset import RSNATrainDataset


def get_ax_without_axes(dpi):
    fig = plt.figure(figsize=(1024 / dpi, 1024 / dpi), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    ax.tick_params(labelbottom=False, bottom=False)  # remove x axis
    ax.tick_params(labelleft=False, left=False)  # remove y axis
    plt.box(on=False)  # turn off the border

    return ax


def main():
    parser = argparse.ArgumentParser(description='Show GT bboxes')
    parser.add_argument('dpi', type=int, help='dpi of your monitor')
    parser.add_argument('--out', '-o', default='results/gt_bboxes')
    args = parser.parse_args()

    dataset = RSNATrainDataset()
    dataset = dataset.slice[23000:, ('patient_id', 'img', 'bbox')]
    print('Total: {} examples.'.format(len(dataset)))

    os.makedirs(args.out, exist_ok=True)
    for i in range(len(dataset)):
        if i % 100 == 0:
            print('Processing {}-th data...'.format(i))
        patient_id, image, bbox = dataset[i]
        image = np.repeat(image, 3, axis=0)

        ax = get_ax_without_axes(args.dpi)
        vis_bbox(image, bbox, np.zeros((len(bbox),)), linewidth=1.0, ax=ax)
        plt.savefig(os.path.join(args.out, '{}.png'.format(patient_id)))
        plt.close()


if __name__ == '__main__':
    main()
