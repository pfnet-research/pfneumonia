import numpy as np
import pandas as pd
from pathlib import Path
import pydicom

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
import rsna

class_names = ('Normal', 'No Lung Opacity / Not Normal', 'Lung Opacity')


class RSNATrainDataset(GetterDataset):

    def __init__(self, base_dir=rsna.DEFAULT_BASE_DIR, omit_negative_samples=False):
        super().__init__()
        self.base_dir = Path(base_dir)

        self.add_getter(('dicom_data', 'img'), self.get_dicom_data_and_image)
        self.add_getter('label', self.get_label)
        self.add_getter('bbox', self.get_bbox)
        self.add_getter('patient_id', self.get_patient_id)
        self.add_getter('view_position', self.get_view_position)

        self.patient_df = pd.read_csv('resources/stage_2_patients_shuffled.csv', index_col=0)
        if omit_negative_samples:
            self.patient_df = self.patient_df[self.patient_df['classIndex'] == 2]

        self.bbox_df = pd.read_csv(self.base_dir / 'stage_2_train_labels.csv')

    def __len__(self):
        return len(self.patient_df)

    def get_patient_id(self, i):
        return self.patient_df['patientId'].values[i]

    def get_index_with_patient_id(self, patient_id):
        """Given patient ID, return patient index for that patient.

        If no such patient ID is found, return -1.
        """
        df = self.patient_df[self.patient_df['patientId'] == patient_id]
        if len(df) == 0:
            return -1
        return df.index.values[0]

    def get_mask_for_positive_samples(self):
        """Mask for positive samples.

        Note that the return value is nonsense if this dataset was initialized with
        `omit_negative_samples=True`.

        Returns:
            mask (~np.ndarray): Array with the same length as `self`. The `i`-th element of `mask`
                is ``True`` if the `i`-th sample is positive (pneumonia); ``False`` otherwise.
        """
        return self.patient_df['classIndex'].values == 2

    def get_dicom_data_and_image(self, i):
        """Getter for DICOM data and image.

        Returns:
            Tuple of (dicom_data, image).
            dicom_data (pydicom.dataset.FileDataset): Raw DICOM data.
            image (~np.ndarray): Greyscale image of shape (1, H, W). All values are within range
                [0, 255].
        """
        patient_id = self.get_patient_id(i)
        dicom_file = str(self.base_dir / 'stage_2_train_images' / '{}.dcm'.format(patient_id))
        dicom_data = pydicom.dcmread(dicom_file)

        image = dicom_data.pixel_array.astype(np.float32)
        image = np.clip(image, 0, 255)
        image = image[np.newaxis]

        return dicom_data, image

    def get_bbox(self, i):
        """Getter for bounding boxes of pneumonia.

        Returns:
            bbox (~np.ndarray): Array of shape (R, 4), where `R` is the number of bboxes on the
                image. Each element represents (y_min, x_min, y_max, x_max).
                If there are no bboxes, array of shape (0, 4) will be returned.
        """
        if self.get_label(i) != 2:
            return np.zeros((0, 4), dtype=np.float32)

        patient_id = self.get_patient_id(i)
        df = self.bbox_df[self.bbox_df['patientId'] == patient_id]
        bbox_values = df[['y', 'x', 'height', 'width']].values
        assert not np.isnan(bbox_values).any()

        bbox = bbox_values.copy().astype(np.float32)
        bbox[:, 2:] += bbox[:, :2]  # (y_min, x_min, y_max, x_max)
        return bbox

    def get_label(self, i):
        """Getter for image-level label.

        Returns:
            label (int): Image-level label, which corresponds to the index of `class_names`.
        """
        patient_id = self.get_patient_id(i)
        df = self.patient_df[self.patient_df['patientId'] == patient_id]
        class_index = df['classIndex'].values[0]
        return class_index

    def get_view_position(self, i):
        return self.get_dicom_data_and_image(i)[0].ViewPosition
