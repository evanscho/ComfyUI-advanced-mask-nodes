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

    ``preserve_interior`` (0.0–1.0) controls how much of the original mask
    interior is kept after blurring.  At 1.0 the interior stays fully opaque,
    giving outward-only feathering.  At 0.0 the blur is unrestricted.
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
                "preserve_interior": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "single_pass": ("BOOLEAN", {"default": False}),
                "soft_cone": ("BOOLEAN", {"default": True}),
                "edge_padding": (
                    ["replicate", "reflect", "zero"],
                    {"default": "replicate"},
                ),
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
        soft_cone: bool = True,
    ) -> tuple:
        """Return a 2-D Gaussian kernel masked to an angular cone.

        Parameters
        ----------
        radius : int
            Blur strength.  Interpreted as sigma (standard deviation) in
            pixels, matching modern Pillow / MaskGrow.  The kernel is
            automatically sized to cover 3σ.
        angle_deg : float
            Centre direction of the blur cone in degrees (0° = up, clockwise).
        spread_deg : float
            Angular width of the cone in degrees.  360° keeps the full kernel.

        Returns
        -------
        kernel : Tensor
            Normalised 2-D kernel.
        kern_radius : int
            Actual half-size of the kernel.
        """
        # σ = radius (matches Pillow ≥11).  Kernel extends to 3σ.
        sigma = float(radius)
        kern_radius = math.ceil(3.0 * sigma)

        size = 2 * kern_radius + 1

        # Co-ordinate grids centred on the kernel middle.
        ax = torch.arange(size, dtype=torch.float32) - kern_radius
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

            if soft_cone:
                # Cosine-tapered angular falloff: full weight in the inner
                # half of the cone, smooth taper to zero in the outer half.
                abs_diff = diff.abs()
                inner_rad = half_spread_rad * 0.5
                angular_mask = torch.where(
                    abs_diff <= inner_rad,
                    torch.ones_like(abs_diff),
                    torch.where(
                        abs_diff <= half_spread_rad,
                        0.5 * (1.0 + torch.cos(
                            math.pi * (abs_diff - inner_rad)
                            / (half_spread_rad - inner_rad)
                        )),
                        torch.zeros_like(abs_diff),
                    ),
                )
            else:
                angular_mask = (diff.abs() <= half_spread_rad).float()

            # Always keep the centre pixel.
            angular_mask[kern_radius, kern_radius] = 1.0

            kernel = kernel * angular_mask

        # Normalise so the kernel sums to 1.
        kernel = kernel / kernel.sum()

        return kernel, kern_radius

    # ------------------------------------------------------------------
    # Single-pass convolution helper
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_kernel(
        mask: torch.Tensor,
        kernel: torch.Tensor,
        radius: int,
        edge_padding: str = "replicate",
    ) -> torch.Tensor:
        """Convolve *mask* (B, H, W) with a prepared (1,1,K,K) kernel."""
        mode = "constant" if edge_padding == "zero" else edge_padding
        pad_args: dict = {"mode": mode}
        if mode == "constant":
            pad_args["value"] = 0.0
        padded = F.pad(
            mask.unsqueeze(1),
            (radius, radius, radius, radius),
            **pad_args,
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
        preserve_interior: float,
        single_pass: bool,
        soft_cone: bool = True,
        edge_padding: str = "replicate",
    ):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # Nothing to do.
        if radius <= 0:
            return (mask,)

        # When single_pass is True, use one kernel of the full radius.
        # Otherwise split into multiple passes to keep memory bounded.
        if single_pass or radius <= self._MAX_SINGLE_RADIUS:
            passes = [radius]
        else:
            # Distribute into equal-ish passes whose variances sum to the
            # target variance:  n * sigma_small^2 = sigma_total^2
            cap = self._MAX_SINGLE_RADIUS
            n_passes = math.ceil(radius / cap)
            # sigma_small = sigma_total / sqrt(n_passes)
            # radius_small = sigma_small * 3
            r_small = max(1, round(radius / math.sqrt(n_passes)))
            r_small = min(r_small, cap)
            passes = [r_small] * n_passes

        result = mask
        for r in passes:
            kernel, kern_radius = self._build_directional_kernel(
                r, angle_deg, spread_deg, soft_cone,
            )
            kernel = kernel.unsqueeze(0).unsqueeze(0).to(mask.device, dtype=mask.dtype)
            result = self._apply_kernel(result, kernel, kern_radius, edge_padding)

        if preserve_interior > 0.0:
            # Blend: at 1.0, fully clamp to original where it was brighter;
            # at fractional values, partially restore the original.
            diff = torch.clamp(mask - result, min=0.0)
            result = result + preserve_interior * diff

        return (result.clamp(0.0, 1.0),)


NODE_CLASS_MAPPINGS = {
    "Directional Mask Blur": DirectionalBlur,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Directional Mask Blur": "Directional Mask Blur",
}
