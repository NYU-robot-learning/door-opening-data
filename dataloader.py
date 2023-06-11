import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import abc

from scipy.spatial.transform import Rotation as R
from PIL import Image
import torchvision
import json


class BaseStickDataset(Dataset, abc.ABC):
    def __init__(self, traj_path, time_skip, time_offset, time_trim):
        super().__init__()
        self.traj_path = Path(traj_path)
        self.time_skip = time_skip
        self.time_offset = time_offset
        self.time_trim = time_trim
        self.img_pth = self.traj_path / "images"
        self.depth_pth = self.traj_path / "depths"
        self.conf_pth = self.traj_path / "confs"
        self.labels_pth = self.traj_path / "labels.json"

        self.labels = json.load(self.labels_pth.open("r"))
        self.img_keys = sorted(self.labels.keys())
        # lable structure: {image_name: {'xyz' : [x,y,z], 'rpy' : [r, p, y], 'gripper': gripper}, ...}

        self.labels = np.array(
            [self.flatten_label(self.labels[k]) for k in self.img_keys]
        )

        # filter using time_skip and time_offset and time_trim. start from time_offset, skip time_skip, and remove last time_trim
        self.labels = self.labels[: -self.time_trim][self.time_offset :: self.time_skip]

        # filter keys using time_skip and time_offset and time_trim. start from time_offset, skip time_skip, and remove last time_trim
        self.img_keys = self.img_keys[: -self.time_trim][
            self.time_offset :: self.time_skip
        ]

    def flatten_label(self, label):
        # flatten label
        xyz = label["xyz"]
        rpy = label["rpy"]
        gripper = label["gripper"]
        return np.concatenate((xyz, rpy, np.array([gripper])))

    def __len__(self):
        return len(self.img_keys)

    def __getitem__(self, idx):
        # not implemented

        raise NotImplementedError


class StickDataset(BaseStickDataset, abc.ABC):
    def __init__(self, traj_path, time_skip, time_offset, time_trim):
        super().__init__(traj_path, time_skip, time_offset, time_trim)
        self.reformat_labels(self.labels)
        self.act_metrics = None

    def set_act_metrics(self, act_metrics):
        self.act_metrics = act_metrics

    def reformat_labels(self, labels):
        # reformat labels to be delta xyz, delta rpy, next gripper state
        new_labels = np.zeros_like(labels)
        new_img_keys = []

        for i in range(len(labels) - 1):
            if i == 0:
                current_label = labels[i]
                next_label = labels[i + 1]
            else:
                next_label = labels[i + 1]

            current_matrix = np.eye(4)
            r = R.from_euler("xyz", current_label[3:6], degrees=False)
            current_matrix[:3, :3] = r.as_matrix()
            current_matrix[:3, 3] = current_label[:3]

            next_matrix = np.eye(4)
            r = R.from_euler("xyz", next_label[3:6], degrees=False)
            next_matrix[:3, :3] = r.as_matrix()
            next_matrix[:3, 3] = next_label[:3]

            delta_matrix = np.linalg.inv(current_matrix) @ next_matrix
            delta_xyz = delta_matrix[:3, 3]
            delta_r = R.from_matrix(delta_matrix[:3, :3])
            delta_rpy = delta_r.as_euler("xyz", degrees=False)

            del_gripper = current_label[6] - next_label[6]
            xyz_norm = np.linalg.norm(delta_xyz)
            rpy_norm = np.linalg.norm(delta_r.as_rotvec())

            if xyz_norm < 0.01 and rpy_norm < 0.008 and abs(del_gripper) < 0.05:
                # drop this label and corresponding image_key since the delta is too small (basically the same image)
                continue

            new_labels[i] = np.concatenate(
                (delta_xyz, delta_rpy, np.array([del_gripper]))
            )
            new_img_keys.append(self.img_keys[i])
            current_label = next_label

        # remove labels with all 0s
        new_labels = new_labels[new_labels.sum(axis=1) != 0]
        assert len(new_labels) == len(new_img_keys)
        self.labels = new_labels
        self.img_keys = new_img_keys

    def load_labels(self, idx):
        # load labels with window size of traj_len, starting from idx and moving window by traj_skip
        labels = self.labels

        return labels

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError()
        return None, self.load_labels(idx)


class ImageStickDataset(StickDataset):
    def __init__(
        self,
        traj_path,
        time_skip,
        time_offset,
        time_trim,
        img_size,
        pre_load=False,
        transforms=None,
    ):
        super().__init__(traj_path, time_skip, time_offset, time_trim)
        self.img_size = img_size
        self.pre_load = pre_load
        self.transforms = transforms
        preprocess_transforms = [torchvision.transforms.ToTensor()]
        if img_size is not None:
            preprocess_transforms = [
                torchvision.transforms.Resize(img_size)
            ] + preprocess_transforms
        self.preprocess_img_transforms = torchvision.transforms.Compose(
            preprocess_transforms
        )

    def __getitem__(self, idx):
        _, labels = super().__getitem__(idx)

        imgs = []
        for key in self.img_keys:
            img = Image.open(str(self.img_pth / key))
            img = self.preprocess_img_transforms(img)
            imgs.append(torch.moveaxis(img, 0, -1))
        # add a nex axis at the beginning
        imgs = torch.stack(imgs, dim=0)

        if self.transforms:
            imgs = self.transforms(imgs)

        return imgs, labels
