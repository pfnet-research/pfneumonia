import numpy as np
import pandas as pd


class PredictionsManager(object):

    def __init__(self):
        self.predictions = {}

    def add_prediction(self, patient_id, bbox, score):
        """Add prediction of a patient.

        Currently, it is not possible to append to or overwrite past predictions.

        Args:
            patient_id (str): Patient ID.
            bbox (~np.ndarray): Bbox of (y_min, x_min, y_max, x_max), shape (R, 4).
            score (~np.ndarray): Confidence score, shape (R,)
        """
        assert len(bbox) == len(score)
        if patient_id in self.predictions:
            raise ValueError('Duplicate Patient ID: {}'.format(patient_id))
        self.predictions[patient_id] = {'bbox': bbox, 'score': score}

    def add_empty_prediction(self, patient_id):
        """Add empty prediction (i.e. no predicted bboxes nor scores) of a patient.

        Args:
            patient_id (str): Patient ID.
        """
        self.add_prediction(patient_id,
                            np.empty((0, 4), dtype=np.float32),
                            np.empty((0,), dtype=np.float32))

    @staticmethod
    def to_prediction_string(one_bbox, one_score):
        """Convert one bbox and score to a prediction string.

        Args:
            one_bbox (~np.ndarray): One bbox, shape (1, 4).
            one_score (float): Scalar confidence score.

        Returns:
            prediction string in format `<score> <x_min> <y_min> <width> <height>`,
            e.g. '0.5 0 0 100 100'.
        """
        y_min, x_min, y_max, x_max = one_bbox
        height, width = y_max - y_min, x_max - x_min
        return '{} {} {} {} {}'.format(one_score, x_min, y_min, width, height)

    def dump(self, f):
        print('patientId,PredictionString', file=f)
        for patient_id, entry in self.predictions.items():
            bbox = entry['bbox']
            score = entry['score']

            prediction_strings = [self.to_prediction_string(bbox[i], score[i])
                                  for i in range(len(bbox))]
            print('{},{}'.format(patient_id, ' '.join(prediction_strings)), file=f)

    def restore(self, f):
        """Restore from a file or buffer.

        Args:
            f: File name or IO buffer.
        """
        df = pd.read_csv(f)
        for row in sorted(df.iterrows()):
            patient_id = row[1]['patientId']
            pred_str = row[1]['PredictionString']
            if row[1].isnull().any():
                values = []
            else:
                values = [float(x) for x in pred_str.split(' ')]
            assert len(values) % 5 == 0
            bboxes = []
            scores = []
            for i in range(len(values) // 5):
                score, x, y, width, height = values[i * 5:i * 5 + 5]
                bboxes.append((y, x, y + height, x + width))
                scores.append(score)
            bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4)
            scores = np.array(scores, dtype=np.float32)
            self.add_prediction(patient_id, bboxes, scores)
