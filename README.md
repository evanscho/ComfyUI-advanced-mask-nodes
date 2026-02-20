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

### Directional Blur

Applies a directional Gaussian blur to a mask, useful for feathering edges in a specific direction.

| Parameter | Description |
|---|---|
| `mask` | Input mask (2D or batched 3D) |
| `angle_deg` | Blur direction in degrees (compass-style: 0° = up, 90° = right, clockwise) |
| `spread_deg` | Angular spread of the blur (180° = half-plane, 360° = omnidirectional) |
| `radius` | Blur radius in pixels |
| `preserve_interior` | When enabled, restores original values inside the mask after blurring |

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
