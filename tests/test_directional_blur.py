"""Smoke tests for the DirectionalBlur node."""
import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from directional_blur import DirectionalBlur


@pytest.fixture
def node():
    return DirectionalBlur()


def test_radius_zero_unchanged(node):
    """radius=0 returns unchanged mask."""
    mask = torch.ones(1, 64, 64)
    (result,) = node.directional_blur(mask, 180.0, 350.0, 0, False)
    assert result.shape == (1, 64, 64)
    assert torch.equal(result, mask)


def test_blur_downward_spread_180(node):
    """White rectangle, blur downward (180°), spread 180 → more blur below than above."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 30:70, 30:70] = 1.0
    (result,) = node.directional_blur(mask, 180.0, 180.0, 10, False)
    assert result.shape == (1, 100, 100)
    bottom_below = result[0, 75, 50].item()
    top_above = result[0, 25, 50].item()
    assert bottom_below > top_above


def test_spread_350_nearly_omni(node):
    """spread=350 blurs in nearly all directions."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 30:70, 30:70] = 1.0
    (result,) = node.directional_blur(mask, 180.0, 350.0, 10, False)
    assert result[0, 25, 50].item() > 0.01
    assert result[0, 75, 50].item() > 0.01


def test_symmetric_interior(node):
    """Symmetric blur dims the interior (no outward-only clamping)."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 30:70, 30:70] = 1.0
    (result,) = node.directional_blur(mask, 180.0, 180.0, 10, False)
    interior_val = result[0, 50, 50].item()
    # Sub-360 kernel sums to <1, so interior dims.
    assert interior_val < 1.0
    assert interior_val > 0.0


def test_spread_360_omnidirectional(node):
    """spread=360 is omnidirectional."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 30:70, 30:70] = 1.0
    (result,) = node.directional_blur(mask, 180.0, 360.0, 10, False)
    assert result[0, 25, 50].item() > 0.01
    assert result[0, 75, 50].item() > 0.01


def test_blur_upward(node):
    """blur upward (0°), spread 180 → more blur above than below."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 30:70, 30:70] = 1.0
    (result,) = node.directional_blur(mask, 0.0, 180.0, 10, False)
    top_above = result[0, 25, 50].item()
    bottom_below = result[0, 75, 50].item()
    assert top_above > bottom_below


def test_no_brightening_narrow_cone(node):
    """Narrowing spread_deg should not brighten beyond the original mask value."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 30:70, 30:70] = 1.0
    (result_90,) = node.directional_blur(mask, 180.0, 90.0, 10, False)
    # Interior should never exceed the original mask value (1.0).
    assert result_90[0, 50, 50].item() <= 1.001


def test_2d_input(node):
    """2D mask input is normalised to 3D output."""
    mask_2d = torch.ones(64, 64)
    (result,) = node.directional_blur(mask_2d, 180.0, 350.0, 5, False)
    assert result.dim() == 3
