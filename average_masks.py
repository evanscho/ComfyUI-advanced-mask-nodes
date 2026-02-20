from typing import Dict

import torch


class AverageOverlappingMasks:
    """Average two masks in regions where they overlap.

    Where both masks have non-zero values the output is the element-wise
    average.  Where only one mask is non-zero the output equals that mask's
    value.  Where neither mask is present the output is zero.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        return {
            "required": {
                "mask_a": ("MASK", {}),
                "mask_b": ("MASK", {}),
                "presence_threshold": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Minimum value for a pixel to be considered 'present' in a mask. "
                               "Raise to ignore soft edges; set to 0 for strict non-zero behavior.",
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "average_masks"
    CATEGORY = "mask"

    def average_masks(self, mask_a: torch.Tensor, mask_b: torch.Tensor,
                      presence_threshold: float = 0.01) -> tuple[torch.Tensor]:
        # Normalise to (B, H, W)
        if mask_a.dim() == 2:
            mask_a = mask_a.unsqueeze(0)
        if mask_b.dim() == 2:
            mask_b = mask_b.unsqueeze(0)

        # Spatial dimensions must match.
        if mask_a.shape[-2:] != mask_b.shape[-2:]:
            raise ValueError(
                f"Spatial dimensions must match: "
                f"mask_a {tuple(mask_a.shape[-2:])} vs "
                f"mask_b {tuple(mask_b.shape[-2:])}"
            )

        # Broadcast along batch dimension if needed.
        if mask_a.shape[0] != mask_b.shape[0]:
            target_b = max(mask_a.shape[0], mask_b.shape[0])
            mask_a = mask_a.expand(target_b, -1, -1)
            mask_b = mask_b.expand(target_b, -1, -1)

        a_present = mask_a > presence_threshold
        b_present = mask_b > presence_threshold
        both = a_present & b_present

        result = torch.zeros_like(mask_a)
        result[both] = (mask_a[both] + mask_b[both]) / 2.0
        result[a_present & ~b_present] = mask_a[a_present & ~b_present]
        result[~a_present & b_present] = mask_b[~a_present & b_present]

        return (result.clamp(0.0, 1.0),)


NODE_CLASS_MAPPINGS = {
    "Average Overlapping Masks": AverageOverlappingMasks,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Average Overlapping Masks": "Average Overlapping Masks",
}
