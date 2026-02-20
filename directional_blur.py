import math
from typing import Dict

import torch
import torch.nn.functional as F


class DirectionalBlur:
    """Applies Gaussian blur to a mask restricted to a configurable angular cone.

    The blur direction is controlled by ``angle_deg`` (compass-style: 0° = up,
    90° = right, 180° = down, 270° = left).  ``spread_deg`` sets how wide the
    blur cone is – 360° gives a standard omnidirectional Gaussian blur while
    smaller values restrict blur to a narrower wedge.

    When ``preserve_interior`` is True the result is clamped with
    ``max(blurred, original)`` so the mask interior stays at 1.0, giving
    outward-only feathering.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        return {
            "required": {
                "mask": ("MASK", {}),
                "angle_deg": (
                    "FLOAT",
                    {"default": 180.0, "min": 0.0, "max": 360.0, "step": 1.0},
                ),
                "spread_deg": (
                    "FLOAT",
                    {"default": 350.0, "min": 10.0, "max": 360.0, "step": 1.0},
                ),
                "radius": (
                    "INT",
                    {"default": 10, "min": 0, "max": 256, "step": 1},
                ),
                "preserve_interior": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "directional_blur"
    CATEGORY = "mask"

    # Maximum single-pass kernel radius.  Larger requested radii are
    # achieved via multiple passes, keeping memory and compute bounded.
    _MAX_SINGLE_RADIUS = 15

    # ------------------------------------------------------------------
    # Kernel helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_directional_kernel(
        radius: int,
        angle_deg: float,
        spread_deg: float,
    ) -> torch.Tensor:
        """Return a 2-D Gaussian kernel masked to an angular cone.

        Parameters
        ----------
        radius : int
            Half-size of the kernel.  Full kernel is ``(2*radius+1)²``.
        angle_deg : float
            Centre direction of the blur cone in degrees (0° = up, clockwise).
        spread_deg : float
            Angular width of the cone in degrees.  360° keeps the full kernel.
        """
        size = 2 * radius + 1
        sigma = radius / 3.0  # 99.7 % of energy within kernel

        # Co-ordinate grids centred on the kernel middle.
        ax = torch.arange(size, dtype=torch.float32) - radius
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")

        # --- Gaussian envelope ---
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))

        # --- Angular mask ---
        if spread_deg < 360.0:
            # In convolution, to blur *toward* a direction we must keep the
            # kernel weights on the *opposite* side (they "reach back" to
            # collect values from the source region).  Negating the
            # direction vector achieves this.
            angle_rad = math.radians(angle_deg % 360.0)
            dir_x = -math.sin(angle_rad)
            dir_y = math.cos(angle_rad)

            # Angle of each kernel pixel relative to centre.
            pixel_angles = torch.atan2(xx, -yy)  # same convention: 0° = up
            direction_angle = math.atan2(dir_x, -dir_y)

            # Angular difference wrapped to [-π, π].
            diff = pixel_angles - direction_angle
            diff = torch.atan2(torch.sin(diff), torch.cos(diff))

            half_spread_rad = math.radians(spread_deg / 2.0)
            angular_mask = (diff.abs() <= half_spread_rad).float()

            # Always keep the centre pixel.
            angular_mask[radius, radius] = 1.0

            kernel = kernel * angular_mask

        # Normalise so the kernel sums to 1.
        kernel = kernel / kernel.sum()
        return kernel

    # ------------------------------------------------------------------
    # Single-pass convolution helper
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_kernel(mask: torch.Tensor, kernel: torch.Tensor, radius: int) -> torch.Tensor:
        """Convolve *mask* (B, H, W) with a prepared (1,1,K,K) kernel."""
        padded = F.pad(
            mask.unsqueeze(1),
            (radius, radius, radius, radius),
            mode="constant",
            value=0.0,
        )
        return F.conv2d(padded, kernel, padding=0).squeeze(1)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def directional_blur(
        self,
        mask: torch.Tensor,
        angle_deg: float,
        spread_deg: float,
        radius: int,
        preserve_interior: bool,
    ):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # Nothing to do.
        if radius <= 0:
            return (mask,)

        # Split into multiple passes when radius exceeds the cap.
        # Each pass uses a smaller kernel; repeated application compounds
        # into the equivalent of a large Gaussian (variances add).
        cap = self._MAX_SINGLE_RADIUS
        if radius <= cap:
            passes = [radius]
        else:
            # Distribute into equal-ish passes whose variances sum to the
            # target variance:  n * sigma_small^2 = sigma_total^2
            n_passes = math.ceil(radius / cap)
            # sigma_small = sigma_total / sqrt(n_passes)
            # radius_small = sigma_small * 3
            r_small = max(1, round(radius / math.sqrt(n_passes)))
            r_small = min(r_small, cap)
            passes = [r_small] * n_passes

        result = mask
        for r in passes:
            kernel = self._build_directional_kernel(r, angle_deg, spread_deg)
            kernel = kernel.unsqueeze(0).unsqueeze(0).to(mask.device, dtype=mask.dtype)
            result = self._apply_kernel(result, kernel, r)

        if preserve_interior:
            result = torch.maximum(result, mask)

        return (result.clamp(0.0, 1.0),)


NODE_CLASS_MAPPINGS = {
    "Directional Blur": DirectionalBlur,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Directional Blur": "Directional Blur",
}
