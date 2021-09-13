
import os
import sys
import warnings
from typing import TYPE_CHECKING, Tuple

import numpy as np
import six

from art import config

if TYPE_CHECKING:
    import torch


DATASET_TYPE = Tuple[  # pylint: disable=C0103
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], float, float
]

def load_cifar10_mj(
    raw: bool = False,
) -> DATASET_TYPE:
    """
    Loads CIFAR10 dataset from config.CIFAR10_PATH or downloads it if necessary.

    :param raw: `True` if no preprocessing should be applied to the data. Otherwise, data is normalized to 1.
    :return: `(x_train, y_train), (x_test, y_test), min, max`
    """

    def load_batch(fpath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Utility function for loading CIFAR batches, as written in Keras.

        :param fpath: Full path to the batch file.
        :return: `(data, labels)`
        """
        with open(fpath, "rb") as file_:
            if sys.version_info < (3,):
                content = six.moves.cPickle.load(file_)
            else:
                content = six.moves.cPickle.load(file_, encoding="bytes")
                content_decoded = {}
                for key, value in content.items():
                    content_decoded[key.decode("utf8")] = value
                content = content_decoded
        data = content["data"]
        labels = content["labels"]

        data = data.reshape(data.shape[0], 3, 32, 32)
        return data, labels

    path = os.path.join('/home/jinie/Documents/cifar-10-batches-py/')
    num_train_samples = 50000

    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype=np.uint8)
    y_train = np.zeros((num_train_samples,), dtype=np.uint8)

    for i in range(1, 6):
        fpath = os.path.join(path, "data_batch_" + str(i))
        data, labels = load_batch(fpath)
        x_train[(i - 1) * 10000 : i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000 : i * 10000] = labels

    fpath = os.path.join(path, "test_batch")
    x_test, y_test = load_batch(fpath)
    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # Set channels last
    x_train = x_train.transpose((0, 2, 3, 1))
    x_test = x_test.transpose((0, 2, 3, 1))

    min_, max_ = 0.0, 255.0
    if not raw:
        min_, max_ = 0.0, 1.0
        x_train, y_train = preprocess(x_train, y_train, clip_values=(0, 255))
        x_test, y_test = preprocess(x_test, y_test, clip_values=(0, 255))

    return (x_train, y_train), (x_test, y_test), min_, max_
