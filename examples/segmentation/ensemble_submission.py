import argparse
import numpy as np

from chainercv.utils import bbox_iou

from rsna.utils.predictions import PredictionsManager


def get_bbox_intersection(bbox_a, bbox_b):
    y_min = max(bbox_a[0], bbox_b[0])
    x_min = max(bbox_a[1], bbox_b[1])
    y_max = min(bbox_a[2], bbox_b[2])
    x_max = min(bbox_a[3], bbox_b[3])
    return np.array((y_min, x_min, y_max, x_max), dtype=np.float32)


def merge_entries(entry1, entry2, thresh):
    bbox1, score1 = entry1['bbox'], entry1['score']
    bbox2, score2 = entry2['bbox'], entry2['score']
    bbox = np.concatenate((bbox1, bbox2), axis=0)
    score = np.concatenate((score1, score2), axis=0)
    if len(score) == 0:
        return bbox, score
    order = score.argsort()[::-1]
    bbox = bbox[order]
    score = score[order]

    iou = bbox_iou(bbox, bbox)
    iou *= 1 - np.eye(len(bbox))  # ignore IoU with itself
    new_bbox = []
    new_score = []
    for i in range(len(bbox)):
        max_iou = iou[i].max()
        if max_iou <= thresh:
            new_bbox.append(bbox[i])
            new_score.append(score[i])
        else:
            max_index = iou[i].argmax()
            if max_index > i:
                new_bbox.append(get_bbox_intersection(bbox[i], bbox[max_index]))
                new_score.append(score[i])
    new_bbox = np.array(new_bbox, dtype=np.float32).reshape(-1, 4)
    new_score = np.array(new_score, dtype=np.float32)
    return new_bbox, new_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input1')
    parser.add_argument('input2')
    parser.add_argument('--out', '-o', default='out.csv')
    parser.add_argument('--thresh', '-t', type=float, default=0.5)
    args = parser.parse_args()

    pred_manager1 = PredictionsManager()
    pred_manager1.restore(args.input1)
    pred_manager2 = PredictionsManager()
    pred_manager2.restore(args.input2)
    assert pred_manager1.predictions.keys() == pred_manager2.predictions.keys()

    new_pred = PredictionsManager()
    n_pos = 0
    for patient_id in pred_manager1.predictions.keys():
        pred1 = pred_manager1.predictions[patient_id]
        pred2 = pred_manager2.predictions[patient_id]
        new_bbox, new_score = merge_entries(pred1, pred2, args.thresh)
        if len(new_bbox) > 0:
            n_pos += 1
        new_pred.add_prediction(patient_id, new_bbox, new_score)

    print('Complete!')
    print('{} / {} are predicted as positive.'.format(n_pos, len(pred_manager1.predictions.keys())))
    with open(args.out, 'w') as f:
        new_pred.dump(f)


if __name__ == '__main__':
    main()
