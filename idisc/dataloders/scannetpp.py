import json
import os
import numpy as np
import torch
from PIL import Image

from .dataset import BaseDataset


class ScannetppNormalsDataset(BaseDataset):

    min_depth = 0.01
    max_depth = 10
    test_split = "proj/data/scannet_utils/meta_data/scannetv2_test.txt"
    train_split = "proj/data/scannet_utils/meta_data/scannetv2_train.txt"

    def __init__(
        self,
        test_mode,
        base_path,
        crop=None,
        benchmark=False,
        augmentations_db={},
        masked=True,
        normalize=True,
        **kwargs,
    ):
        super().__init__(test_mode, base_path, benchmark, normalize)
        self.test_mode = test_mode
        self.crop = crop
        self.height = 584
        self.width = 876
        self.masked = masked

        # load annotations
        self.load_dataset()
        for k, v in augmentations_db.items():
            setattr(self, k, v)

    def load_dataset(self):
        self.invalid_depth_num = 0

        image_path = os.path.join(self.base_path, "data")
        depth_path = os.path.join(self.base_path, "depths")
        normal_path = os.path.join(self.base_path, "normals")

        all_scenes = os.listdir(image_path)
        if not self.test_mode:
            all_scenes = all_scenes[:-50]
        else:
            all_scenes = all_scenes[-50:]

        for scene in all_scenes:
            scene_image_path = os.path.join(image_path, scene, "dslr", "resized_images_2")
            scene_depth_path = os.path.join(depth_path, scene)
            scene_normal_path = os.path.join(normal_path, scene)
            image_files = os.listdir(scene_image_path)
            for image_file in image_files:
                img_info = dict(scene_id=scene)
                if not self.benchmark:
                    depth_map = os.path.join(scene_depth_path, image_file.replace(".JPG", "bin"))
                    normal_map = os.path.join(scene_normal_path, image_file.replace(".JPG", "jpg"))
                    img_info["annotation_filename_depth"] = depth_map
                    img_info["annotation_filename_normals"] = normal_map
                image = os.path.join(scene_image_path, image_file)
                img_info["image_filename"] = image
                self.dataset.append(img_info)
        print(
            f"Loaded {len(self.dataset)} images. Totally {self.invalid_depth_num} invalid pairs are filtered"
        )

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        image = np.asarray(
            Image.open(self.dataset[idx]["image_filename"]))
        normals = np.asarray(
            Image.open(self.dataset[idx]["annotation_filename_normals"])).astype(np.uint8)[..., :3]
        info = self.dataset[idx].copy()
        scene_id = info["scene_id"]
        transform = json.load(open(os.path.join(self.base_path, "data", scene_id, "dslr", "nerfstudio", "transforms_2.json", "r")))
        fl_x, fl_y, cx, cy = transform['fl_x'], transform['fl_y'], transform['cx'], transform['cy']
        intrin = torch.from_numpy(np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]]))
        info["camera_intrinsics"] = intrin
        image, gts, info = self.transform(image=image,
                                          gts={"normals": normals},
                                          info=info)
        return {"image": image, "gt": gts["gt"], "mask": gts["mask"]}

    def get_pointcloud_mask(self, shape):
        mask = np.zeros(shape)
        height_start, height_end = 45, self.height - 9
        width_start, width_end = 41, self.width - 39
        mask[height_start:height_end, width_start:width_end] = 1
        return mask

    def preprocess_crop(self, image, gts=None, info=None):
        height_start, height_end = 0, self.height
        width_start, width_end = 0, self.width
        image = image[height_start:height_end, width_start:width_end]
        new_gts = {}
        if "normals" in gts:
            normals = gts["normals"]
            mask = (normals.sum(axis=-1) > 0).astype(np.uint8)
            new_gts["gt"] = normals
            new_gts["mask"] = mask

        info["camera_intrinsics"][0, 2] = info["camera_intrinsics"][0, 2] - width_start
        info["camera_intrinsics"][1, 2] = info["camera_intrinsics"][1, 2] - height_start
        return image, new_gts, info

    def eval_mask(self, valid_mask):
        """Do grag_crop or eigen_crop for testing"""

        border_mask = np.zeros_like(valid_mask)
        border_mask[45:584-9, 41:876-39] = 1
        return np.logical_and(valid_mask, border_mask)
