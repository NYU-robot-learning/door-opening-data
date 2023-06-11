import itertools
from torch.utils.data import ConcatDataset
from pathlib import Path
import numpy as np

from dataloader import ImageStickDataset
from traverse_data import iter_dir_for_traj_pths


def get_image_stick_dataset_single(
    data_path,
    time_skip=4,
    time_offset=5,
    time_trim=5,
    img_size=224,
):
    # add transforms for normalization and converting to float tensor
    if type(data_path) == str:
        data_path = Path(data_path)

    val_mask = {"home": [], "env": [], "traj": []}
    mask_texts = []
    train_traj_paths, _, _ = iter_dir_for_traj_pths(data_path, val_mask, mask_texts)

    train_dataset = ConcatDataset(
        [
            ImageStickDataset(
                traj_path,
                time_skip,
                time_offset,
                time_trim,
                img_size,
                pre_load=False,
                transforms=None,
            )
            for traj_path in train_traj_paths
        ]
    )

    return train_dataset


def get_image_stick_dataset_double(
    data_path,
    time_skip=4,
    time_offset=5,
    time_trim=5,
    img_size=224,
):
    # add transforms for normalization and converting to float tensor
    if type(data_path) == str:
        data_path = Path(data_path)

    val_mask = {"home": [], "env": [], "traj": []}
    mask_texts = []
    train_traj_paths, _, _ = iter_dir_for_traj_pths(data_path, val_mask, mask_texts)

    train_dataset = ConcatDataset(
        [
            ImageStickDataset(
                traj_path,
                time_skip,
                time_offset_n,
                time_trim,
                img_size,
                pre_load=False,
                transforms=None,
            )
            for traj_path, time_offset_n in itertools.product(
                train_traj_paths, [time_offset, time_offset + 2]
            )
        ]
    )

    return train_dataset


if __name__ == "__main__":
    # the following code loads the dataset. In this we sample each expert demonstration twice.
    # So if there are 10 demonstrations, we have 20 trajectories in the dataset.
    # This done by sampling each demo by 2 different time offsets but similar time skips.
    # If this is not desired, use get_image_stick_dataset_single instead.

    dataset = get_image_stick_dataset_double(
        "./CDS_Home",
        time_skip=4,  # number of frames to skip for each state in a trajectory
        time_offset=5,  # intial frames to crop out (to reduce noise)
        time_trim=5,  # final frames to crop out (to reduce noise)
        img_size=None,  # with None, images are loaded at original size. Can be set to 224 for resnet
    )

    array_of_trajectories = np.asarray([])
    task = np.asarray(["door opening"])

    for i in range(20):
        dat = dataset[i]
        trajectory = {
            "action": dat[1],
            "image": dat[0].numpy(),
            "task": task,
        }
        array_of_trajectories = np.append(array_of_trajectories, trajectory)

    np.save("./door_opening_data.npy", array_of_trajectories)
