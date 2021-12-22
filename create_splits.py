import argparse
import glob
import os
import random
import shutil

import numpy as np

from utils import get_module_logger

TRAIN_PERCENTAGE = 0.7
VAL_PERCENTAGE = 0.15
TEST_PERCENTAGE = 0.15
SEED = 42


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """

    if not os.path.isdir(source):
        print("Source directory {} does not exists!".format(source))
        return

    if not os.path.isdir(destination):
        print("Destination directory {} does not exists!".format(destination))
        return

    dataset_dirs = [
        os.path.join(destination, "train"),
        os.path.join(destination, "val"),
        os.path.join(destination, "test"),
    ]

    all_tfrecords = glob.glob("{}/*.tfrecord".format(source))

    np.random.seed(SEED)
    np.random.shuffle(all_tfrecords)

    dataset_size = len(all_tfrecords)
    train_size = int(TRAIN_PERCENTAGE * dataset_size)
    val_size = int(VAL_PERCENTAGE * dataset_size)
    test_size = int(TEST_PERCENTAGE * dataset_size)

    dataset = np.split(
        all_tfrecords,
        [train_size, train_size + val_size],
    )

    for data, directory in zip(dataset, dataset_dirs):
        os.makedirs(directory, exist_ok=True)

        for tfrecord in data:
            print(
                "Creating symplink for file {} into {} ...".format(tfrecord, directory)
            )
            #os.symlink(tfrecord, os.path.join(directory, os.path.basename(tfrecord)))
            shutil.copy(tfrecord, os.path.join(directory, os.path.basename(tfrecord)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split data into training / validation / testing"
    )
    parser.add_argument("--source", required=True, help="source data directory")
    parser.add_argument(
        "--destination", required=True, help="destination data directory"
    )
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info("Creating splits...")
    split(args.source, args.destination)
