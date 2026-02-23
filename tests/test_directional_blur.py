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
    (result,) = node.directional_blur(mask, 180.0, 350.0, 0, 1.0, False)
    assert result.shape == (1, 64, 64)
    assert torch.equal(result, mask)


def test_blur_downward_spread_180(node):
    """White rectangle, blur downward (180°), spread 180 → blurs below, sharp above."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 30:70, 30:70] = 1.0
    (result,) = node.directional_blur(mask, 180.0, 180.0, 10, 0.0, False)
    assert result.shape == (1, 100, 100)
    bottom_below = result[0, 75, 50].item()
    assert 0.0 < bottom_below < 1.0
    top_above = result[0, 25, 50].item()
    assert top_above < 0.01


def test_spread_350_nearly_omni(node):
    """spread=350 blurs in nearly all directions."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 30:70, 30:70] = 1.0
    (result,) = node.directional_blur(mask, 180.0, 350.0, 10, 0.0, False)
    assert result[0, 25, 50].item() > 0.01
    assert result[0, 75, 50].item() > 0.01


def test_preserve_interior(node):
    """preserve_interior keeps interior at ~1.0."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 30:70, 30:70] = 1.0
    (result,) = node.directional_blur(mask, 180.0, 180.0, 10, 1.0, False)
    interior_min = result[0, 40:60, 40:60].min().item()
    assert interior_min >= 0.999


def test_spread_360_omnidirectional(node):
    """spread=360 is omnidirectional."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 30:70, 30:70] = 1.0
    (result,) = node.directional_blur(mask, 180.0, 360.0, 10, 0.0, False)
    assert result[0, 25, 50].item() > 0.01
    assert result[0, 75, 50].item() > 0.01


def test_blur_upward(node):
    """blur upward (0°), spread 180 → blurs above, sharp below."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 30:70, 30:70] = 1.0
    (result,) = node.directional_blur(mask, 0.0, 180.0, 10, 0.0, False)
    assert result[0, 25, 50].item() > 0.01
    assert result[0, 75, 50].item() < 0.01


def test_2d_input(node):
    """2D mask input is normalised to 3D output."""
    mask_2d = torch.ones(64, 64)
    (result,) = node.directional_blur(mask_2d, 180.0, 350.0, 5, 1.0, False)
    assert result.dim() == 3
