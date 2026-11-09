import random
import numpy as np
import torch


class PadTo256:
    """Zero-pads image on bottom and right to 256x256, padding target with 255."""

    def __init__(self, target_size: int = 256, pad_value_mask: int = 255):
        self.target_size = target_size
        self.pad_value_mask = pad_value_mask

    def __call__(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # image: (2, H, W), mask: (H, W)
        _, h, w = image.shape
        if h == self.target_size and w == self.target_size:
            return image, mask

        pad_h = max(0, self.target_size - h)
        pad_w = max(0, self.target_size - w)

        if pad_h > 0 or pad_w > 0:
            padded_image = np.pad(
                image,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0.0,
            )
            padded_mask = np.pad(
                mask,
                ((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=self.pad_value_mask,
            )
            return padded_image[:, :self.target_size, :self.target_size], padded_mask[:self.target_size, :self.target_size]

        return image[:, :self.target_size, :self.target_size], mask[:self.target_size, :self.target_size]


class NormalizeSAR:
    """
    Clips VV and VH channels to training-set [p1, p99] percentiles and maps to [-1, 1].
    Sets invalid pixels (where mask == 255 or non-finite values) to 0.0.
    """

    def __init__(self, vv_limits: tuple[float, float], vh_limits: tuple[float, float]):
        self.vv_min, self.vv_max = vv_limits
        self.vh_min, self.vh_max = vh_limits

    def __call__(self, image: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
        image = image.astype(np.float32)

        # Non-finite values mapped to 0
        non_finite = ~np.isfinite(image)
        image[non_finite] = 0.0

        # Channel 0: VV
        vv_denom = max(self.vv_max - self.vv_min, 1e-6)
        vv_clipped = np.clip(image[0], self.vv_min, self.vv_max)
        image[0] = 2.0 * ((vv_clipped - self.vv_min) / vv_denom) - 1.0

        # Channel 1: VH
        vh_denom = max(self.vh_max - self.vh_min, 1e-6)
        vh_clipped = np.clip(image[1], self.vh_min, self.vh_max)
        image[1] = 2.0 * ((vh_clipped - self.vh_min) / vh_denom) - 1.0

        if valid_mask is not None:
            image[:, ~valid_mask] = 0.0

        return image


class RandomSARAugmentation:
    """
    Applies identical random spatial transformations to image and mask:
      - Horizontal flip (p=0.5)
      - Vertical flip (p=0.5)
      - Rotation from {0, 90, 180, 270} degrees
    """

    def __call__(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # image: (2, H, W), mask: (H, W)
        if random.random() < 0.5:
            image = np.flip(image, axis=2)
            mask = np.flip(mask, axis=1)

        if random.random() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=0)

        rot_k = random.choice([0, 1, 2, 3])
        if rot_k > 0:
            image = np.rot90(image, k=rot_k, axes=(1, 2))
            mask = np.rot90(mask, k=rot_k, axes=(0, 1))

        return np.ascontiguousarray(image), np.ascontiguousarray(mask)
