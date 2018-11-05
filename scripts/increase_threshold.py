import argparse

from rsna.utils import PredictionsManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='out.csv')
    parser.add_argument('--out', '-o')
    parser.add_argument('--thresh', '-t', type=float, required=True)
    args = parser.parse_args()

    if args.out is None:
        args.out = 't{}-{}'.format(args.thresh, args.input)

    pred = PredictionsManager()
    pred.restore(args.input)
    new_pred = PredictionsManager()

    n_positive = 0
    n_cut = 0
    for patient_id, entry in pred.predictions.items():
        bbox, score = entry['bbox'], entry['score']
        mask = score >= args.thresh
        if mask.sum() > 0:
            n_positive += 1
        elif len(score) > 0:
            n_cut += 1
        bbox = bbox[mask]
        score = score[mask]
        new_pred.add_prediction(patient_id, bbox, score)

    with open(args.out, 'w') as f:
        new_pred.dump(f)

    print('Complete with {} positives.'.format(n_positive))
    if n_cut == 0:
        print('WARNINGS: no predictions were cut. Check the threshold.')


if __name__ == '__main__':
    main()
