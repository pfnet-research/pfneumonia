import argparse
import os

from rsna.utils.predictions import PredictionsManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', default='cross-validation',
                        help='Path to cross-validation directory')
    parser.add_argument('--filename', '-f', default='t0.01.csv',
                        help='File name for each CSV file')
    parser.add_argument('--out', '-o', default='merged.csv')
    args = parser.parse_args()

    pred = PredictionsManager()
    for i in range(10):
        pred.restore(os.path.join(args.dir, 'cv{}'.format(i), args.filename))
    with open(args.out, 'w') as f:
        pred.dump(f)


if __name__ == '__main__':
    main()
