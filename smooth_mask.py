from typing import Dict

import torch
import torch.nn.functional as F


class SmoothMask:
    """Iterative Laplacian smoothing for masks.

    Each iteration, every pixel moves toward the average of its
    neighbours.  Sharp steps between adjacent pixels get softened
    while already-smooth gradients are barely affected.

    This is useful for cleaning up hard edges or banding artefacts
    in mask gradients without re-blurring the whole mask.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        return {
            "required": {
                "mask": ("MASK", {}),
                "iterations": (
                    "INT",
                    {"default": 5, "min": 1, "max": 100, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "smooth_mask"
    CATEGORY = "mask"

    # 3×3 Laplacian averaging kernel: centre pixel = average of its
    # 8 neighbours + itself (uniform box filter).
    _KERNEL = torch.tensor(
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],
        dtype=torch.float32,
    ).unsqueeze(0).unsqueeze(0) / 9.0

    def smooth_mask(
        self,
        mask: torch.Tensor,
        iterations: int,
    ):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        result = mask
        kernel = self._KERNEL.to(mask.device, dtype=mask.dtype)

        for _ in range(iterations):
            padded = F.pad(
                result.unsqueeze(1),
                (1, 1, 1, 1),
                mode="replicate",
            )
            result = F.conv2d(padded, kernel, padding=0).squeeze(1)

        return (result.clamp(0.0, 1.0),)


NODE_CLASS_MAPPINGS = {
    "Smooth Mask": SmoothMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Smooth Mask": "Smooth Mask",
}
