import numpy as np
import pandas as pd

from rsna.datasets.rsna_train_dataset import class_names


src_csv = './RSNA/stage_2_detailed_class_info.csv'
dst_csv = './resources/stage_2_patients_shuffled.csv'


def main(seed):
    df = pd.read_csv(src_csv)
    df = df.drop_duplicates('patientId')
    df = df.sort_values('patientId')

    np.random.seed(seed)
    order = np.arange(len(df))
    np.random.shuffle(order)
    df['patientIndex'] = order
    df = df.sort_values('patientIndex')

    # Convert 'class' to numerical indices
    df['classIndex'] = df['class'].apply(lambda x: class_names.index(x))

    # Those images whose IDs are similar to test ones
    former = ('000924cf-0f8d-42bd-9158-1af53881a557' < df['patientId']) & (df['patientId'] < '313677e1-a894-4abc-814c-42ce8ab44dde')
    latter = ('bfea966c-f72f-42f7-9712-9cf77b66bad6' < df['patientId']) & (df['patientId'] < 'c1f7889a-9ea9-4acb-b64c-b737c929599a')
    df['withinTestRange'] = (former | latter).map(int)

    df.to_csv(dst_csv, columns=('patientIndex', 'patientId', 'classIndex', 'withinTestRange'),
              index=False)


if __name__ == '__main__':
    main(0)
