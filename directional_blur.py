import math
from typing import Dict

import torch
import torch.nn.functional as F


class DirectionalBlur:
    """Directional mask blur via gradient blending.

    Computes a full omnidirectional Gaussian blur (identical to MaskGrow /
    PIL ``GaussianBlur``), then blends between the blurred result and the
    original mask using a per-pixel weight that varies along the direction
    specified by ``angle_deg``.

    The side of the mask facing ``angle_deg`` receives full blur while the
    opposite side stays sharp, with a smooth gradient between.

    Two blend modes are available:

    * **bbox** – projects each pixel onto the ``angle_deg`` axis relative
      to the mask's bounding-box centre and normalises by the bounding-box
      extent.  Fast and simple.

    * **sdf** – uses a signed-distance field from the mask boundary to
      weight the blend.  Each boundary pixel's position is projected onto
      the ``angle_deg`` axis, producing a weight map that adapts precisely
      to the mask shape.  More expensive but handles irregular masks
      better.
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
                "radius": (
                    "INT",
                    {"default": 10, "min": 0, "max": 256, "step": 1},
                ),
                "blend_mode": (
                    ["bbox", "sdf"],
                    {"default": "bbox"},
                ),
                "falloff": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1},
                ),
                "single_pass": ("BOOLEAN", {"default": False}),
                "edge_padding": (
                    ["replicate", "reflect", "zero"],
                    {"default": "replicate"},
                ),
                "sdf_iterations": (
                    "INT",
                    {"default": 30, "min": 5, "max": 200, "step": 5},
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "directional_blur"
    CATEGORY = "mask"

    _MAX_SINGLE_RADIUS = 15

    # ------------------------------------------------------------------
    # Gaussian blur helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_gaussian_kernel(radius: int) -> tuple:
        """Return an omnidirectional normalised 2-D Gaussian kernel."""
        sigma = float(radius)
        kern_radius = math.ceil(3.0 * sigma)

        size = 2 * kern_radius + 1

        # Co-ordinate grids centred on the kernel middle.
        ax = torch.arange(size, dtype=torch.float32) - kern_radius
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")

        # --- Gaussian envelope, normalised to 1.0 ---
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel, kern_radius

    @staticmethod
    def _apply_kernel(
        mask: torch.Tensor,
        kernel: torch.Tensor,
        radius: int,
        edge_padding: str = "replicate",
    ) -> torch.Tensor:
        """Convolve *mask* (B, H, W) with a (1,1,K,K) kernel."""
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
    # Blend-weight helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _blend_weight_bbox(
        mask: torch.Tensor,
        angle_deg: float,
    ) -> torch.Tensor:
        """Per-pixel blend weight based on bounding-box projection.

        For each batch element, projects pixel positions onto the
        ``angle_deg`` axis relative to the mask's bounding-box centre,
        normalised by the mask extent along that axis.  Returns a weight
        map in [0, 1] where 1 = fully blurred (leading edge) and 0 =
        original (trailing edge).
        """
        B, H, W = mask.shape
        angle_rad = math.radians(angle_deg % 360.0)
        # Direction vector (0° = up → dy=-1, dx=0).
        dx = math.sin(angle_rad)
        dy = -math.cos(angle_rad)

        # Pixel coordinate grids.
        ys = torch.arange(H, dtype=mask.dtype, device=mask.device)
        xs = torch.arange(W, dtype=mask.dtype, device=mask.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        weight = torch.zeros_like(mask)

        for b in range(B):
            # Find bounding box of non-zero pixels.
            nz = mask[b] > 0.01
            if not nz.any():
                continue

            nz_y, nz_x = torch.where(nz)
            cy = (nz_y.float().min() + nz_y.float().max()) / 2.0
            cx = (nz_x.float().min() + nz_x.float().max()) / 2.0

            # Project each pixel position onto the direction axis,
            # relative to the bounding-box centre.
            proj = (grid_x - cx) * dx + (grid_y - cy) * dy

            # Project all mask pixels to find the extent.
            mask_proj = (nz_x.float() - cx) * dx + (nz_y.float() - cy) * dy
            proj_min = mask_proj.min()
            proj_max = mask_proj.max()
            extent = proj_max - proj_min
            if extent < 1.0:
                extent = 1.0

            # Normalise: leading edge (max projection) → 1, trailing → 0.
            w = (proj - proj_min) / extent
            weight[b] = w.clamp(0.0, 1.0)

        return weight

    @staticmethod
    def _blend_weight_sdf(
        mask: torch.Tensor,
        angle_deg: float,
        blur_radius: int,
        sdf_iterations: int = 30,
    ) -> torch.Tensor:
        """Per-pixel blend weight based on a signed-distance field.

        Computes the distance from each pixel to the nearest mask
        boundary pixel, then weights by the boundary pixel's position
        projected onto the ``angle_deg`` axis.  Pixels near boundary
        regions that face the angle get weight → 1 (blurred), those near
        the opposite boundary get weight → 0 (original).
        """
        B, H, W = mask.shape
        angle_rad = math.radians(angle_deg % 360.0)
        dx = math.sin(angle_rad)
        dy = -math.cos(angle_rad)

        weight = torch.zeros_like(mask)

        for b in range(B):
            m = mask[b]

            # Find boundary pixels via dilation - erosion.
            m_4d = m.unsqueeze(0).unsqueeze(0)
            dilated = F.max_pool2d(m_4d, kernel_size=3, stride=1, padding=1)
            eroded = -F.max_pool2d(-m_4d, kernel_size=3, stride=1, padding=1)
            boundary = ((dilated - eroded) > 0.01).squeeze()

            if not boundary.any():
                # No boundary → uniform mask, weight = 0.5 everywhere.
                weight[b] = 0.5
                continue

            bnd_y, bnd_x = torch.where(boundary)
            bnd_y = bnd_y.float()
            bnd_x = bnd_x.float()

            # Centre of the boundary.
            cy = (bnd_y.min() + bnd_y.max()) / 2.0
            cx = (bnd_x.min() + bnd_x.max()) / 2.0

            # Project boundary pixels onto the angle axis.
            bnd_proj = (bnd_x - cx) * dx + (bnd_y - cy) * dy
            proj_min = bnd_proj.min()
            proj_max = bnd_proj.max()
            extent = proj_max - proj_min
            if extent < 1.0:
                extent = 1.0

            # Normalise boundary projections to [0, 1].
            bnd_weight = ((bnd_proj - proj_min) / extent).clamp(0.0, 1.0)

            # For each pixel, find the nearest boundary pixel and use
            # that boundary pixel's directional weight.
            # To keep this tractable, use a distance-transform-like
            # approach: iteratively propagate boundary weights outward.
            # sdf_iterations controls how far the propagation reaches:
            # low values → only near-boundary pixels get directional
            # blending; high values → full propagation across the mask.
            spread = sdf_iterations
            w = torch.full((H, W), 0.5, dtype=mask.dtype, device=mask.device)
            dist = torch.full((H, W), float("inf"), dtype=mask.dtype, device=mask.device)

            # Seed boundary pixels.
            dist[boundary] = 0.0
            w[boundary] = bnd_weight.to(mask.dtype)

            # Iterative propagation (like a BFS distance transform).
            for _ in range(spread):
                # Pad dist and w for 3x3 neighbourhood.
                d_pad = F.pad(
                    dist.unsqueeze(0).unsqueeze(0),
                    (1, 1, 1, 1),
                    mode="replicate",
                ).squeeze()
                w_pad = F.pad(
                    w.unsqueeze(0).unsqueeze(0),
                    (1, 1, 1, 1),
                    mode="replicate",
                ).squeeze()

                # Check all 8 neighbours + self.
                for oy in range(-1, 2):
                    for ox in range(-1, 2):
                        if oy == 0 and ox == 0:
                            continue
                        step = math.sqrt(oy * oy + ox * ox)
                        nd = d_pad[1 + oy:H + 1 + oy, 1 + ox:W + 1 + ox] + step
                        nw = w_pad[1 + oy:H + 1 + oy, 1 + ox:W + 1 + ox]
                        closer = nd < dist
                        dist = torch.where(closer, nd, dist)
                        w = torch.where(closer, nw, w)

            weight[b] = w.clamp(0.0, 1.0)

        # Smooth the weight map to eliminate Voronoi seams from the
        # nearest-boundary propagation.  Use a Gaussian whose sigma
        # scales with blur_radius so the smoothing is proportional
        # to the blur extent.
        smooth_sigma = max(1.0, blur_radius * 0.5)
        smooth_kr = math.ceil(3.0 * smooth_sigma)
        smooth_size = 2 * smooth_kr + 1
        ax = torch.arange(smooth_size, dtype=weight.dtype, device=weight.device) - smooth_kr
        sy, sx = torch.meshgrid(ax, ax, indexing="ij")
        smooth_kern = torch.exp(-(sx ** 2 + sy ** 2) / (2.0 * smooth_sigma ** 2))
        smooth_kern = smooth_kern / smooth_kern.sum()
        smooth_kern = smooth_kern.unsqueeze(0).unsqueeze(0)

        w_4d = F.pad(
            weight.unsqueeze(1),
            (smooth_kr, smooth_kr, smooth_kr, smooth_kr),
            mode="replicate",
        )
        weight = F.conv2d(w_4d, smooth_kern, padding=0).squeeze(1)
        weight = weight.clamp(0.0, 1.0)

        return weight

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def directional_blur(
        self,
        mask: torch.Tensor,
        angle_deg: float,
        radius: int,
        blend_mode: str = "bbox",
        falloff: float = 1.0,
        single_pass: bool = False,
        edge_padding: str = "replicate",
        sdf_iterations: int = 30,
    ):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

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

        # --- Full omnidirectional Gaussian blur ---
        blurred = mask
        for r in passes:
            kernel, kern_radius = self._build_gaussian_kernel(r)
            kernel = kernel.unsqueeze(0).unsqueeze(0).to(
                mask.device, dtype=mask.dtype,
            )
            blurred = self._apply_kernel(
                blurred, kernel, kern_radius, edge_padding,
            )

        # --- Directional blend weight ---
        if blend_mode == "sdf":
            weight = self._blend_weight_sdf(mask, angle_deg, radius, sdf_iterations)
        else:
            weight = self._blend_weight_bbox(mask, angle_deg)

        # Apply falloff curve: >1 steepens (blur concentrates at
        # leading edge), <1 flattens (blur extends toward trailing).
        if falloff != 1.0:
            weight = weight.pow(falloff)

        # Blend: weight=1 → blurred, weight=0 → original.
        result = weight * blurred + (1.0 - weight) * mask

        return (result.clamp(0.0, 1.0),)


NODE_CLASS_MAPPINGS = {
    "Directional Mask Blur": DirectionalBlur,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Directional Mask Blur": "Directional Mask Blur",
}
