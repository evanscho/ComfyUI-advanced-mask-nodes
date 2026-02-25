# Advanced Mask Nodes

Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that provide advanced mask manipulation operations.

## Nodes

### Average Overlapping Masks

Averages two masks in regions where they overlap.

- **Overlap regions** — output is the element-wise average of both masks
- **Single-mask regions** — output equals that mask's value
- **Empty regions** — output is zero

| Parameter | Description |
|---|---|
| `mask_a` / `mask_b` | Input masks (2D or batched 3D) |
| `presence_threshold` | Minimum value to consider a pixel "present" (default `0.01`). Raise to ignore soft edges; set to `0` for strict non-zero behavior. |
| `blend_threshold` | Minimum value for a pixel to participate in averaging (default `0.01`). Pixels between `presence_threshold` and `blend_threshold` are present but passed through without blending. |

### Directional Mask Blur

Computes a full omnidirectional Gaussian blur (identical to MaskGrow), then blends between the blurred result and the original mask using a per-pixel directional weight. The side of the mask facing `angle_deg` receives full blur while the opposite side stays sharp.

| Parameter | Description |
|---|---|
| `mask` | Input mask (2D or batched 3D) |
| `angle_deg` | Blur direction in degrees (compass-style: 0° = up, 90° = right, clockwise) |
| `radius` | Blur radius in pixels (sigma for the Gaussian kernel) |
| `blend_mode` | How the directional weight map is computed: `bbox` (fast, bounding-box based) or `sdf` (precise, signed-distance field based) |
| `falloff` | Controls the steepness of the directional gradient (default `1.0`). Values `< 1` spread blur further toward the trailing edge; values `> 1` concentrate it near the leading edge. |
| `single_pass` | When enabled, uses a single large kernel instead of multi-pass |
| `edge_padding` | Padding mode for convolution edges: `replicate`, `reflect`, or `zero` |
| `sdf_iterations` | Number of propagation iterations for SDF mode (default `30`). Higher = more precise for large or irregular masks. |

### Smooth Mask

Iterative Laplacian smoothing for masks. Each iteration, every pixel moves toward the average of its neighbours. Sharp steps get softened while already-smooth gradients are barely affected.

| Parameter | Description |
|---|---|
| `mask` | Input mask (2D or batched 3D) |
| `iterations` | Number of smoothing passes (default `5`). More iterations = smoother result. |
| `only_increase` | When enabled, smoothing can only increase pixel values — dips get filled in but peaks are never lowered. Useful for filling gaps without losing coverage. |

## Installation

Clone into your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/evanscho/advanced_mask_nodes
```

Or install via [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager).

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .
```

## License

[GNU General Public License v3](LICENSE)
