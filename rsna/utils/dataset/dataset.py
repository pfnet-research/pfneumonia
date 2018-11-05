import numpy as np

from chainercv.chainer_experimental.datasets.sliceable import ConcatenatedDataset
import chainermn


def oversample_dataset(dataset, mask, rate):
    """Perform oversampling.

    Args:
        dataset (SliceableDataset): Source dataset.
        mask (~numpy.ndarray): `mask[i]` must be `True` iff `i`-th sample of `dataset` is a target
            of oversampling.
        rate (int): Target samples will appear `rate` times in the dataset after oversampling.

    Returns:
        ConcatenatedDataset: Dataset after oversampling. Its elements will not be shuffled.
    """
    target_data = dataset.slice[mask.nonzero()[0]]
    datasets = [dataset]

    for _ in range(rate - 1):
        datasets.append(target_data)
    return ConcatenatedDataset(*datasets)


def scatter_train_and_val_data(train_data, val_data, comm):
    if comm.rank == 0:
        indices = np.arange(len(train_data))
    else:
        indices = None
    indices = chainermn.scatter_dataset(indices, comm, shuffle=True)
    train_data = train_data.slice[indices]

    indices = np.arange(len(val_data))
    indices = indices[comm.rank: indices.size: comm.size]
    val_data = val_data.slice[indices]

    return train_data, val_data
