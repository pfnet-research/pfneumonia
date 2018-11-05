from chainer.dataset import DatasetMixin
import numpy as np
from pathlib import Path
import pydicom

import rsna


class RSNASubmissionDataset(DatasetMixin):

    def __init__(self, base_dir=rsna.DEFAULT_BASE_DIR):
        super().__init__()
        self.base_dir = Path(base_dir)

        test_image_dir = self.base_dir / 'stage_2_test_images'
        dicom_files = test_image_dir.glob('*.dcm')
        self.patient_ids = np.array(sorted([x.stem for x in dicom_files]), dtype=object)

    def __len__(self):
        return len(self.patient_ids)

    def get_index_with_patient_id(self, patient_id):
        return self.patient_ids.searchsorted(patient_id)

    def get_dicom_data_and_image(self, i):
        return self.get_example(i)

    def get_example(self, i):
        """Load `i`-th example of test dataset.

        Returns:
            Tuple of (dicom_data, image).
            dicom_data (pydicom.dataset.FileDataset): Raw DICOM data.
            image (~np.ndarray): Greyscale image of shape (1, H, W). All values are within range
                [0, 255].
        """
        patient_id = self.patient_ids[i]

        dicom_file = str(self.base_dir / 'stage_2_test_images' / '{}.dcm'.format(patient_id))
        dicom_data = pydicom.dcmread(dicom_file)
        image = dicom_data.pixel_array.astype(np.float32)
        image = np.clip(image, 0, 255)
        image = image[np.newaxis]

        return dicom_data, image
