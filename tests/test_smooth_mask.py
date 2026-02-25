"""Tests for the SmoothMask node."""
import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from smooth_mask import SmoothMask


@pytest.fixture
def node():
    return SmoothMask()


def test_uniform_mask_unchanged(node):
    """A uniform mask should stay uniform after smoothing."""
    mask = torch.ones(1, 64, 64) * 0.5
    (result,) = node.smooth_mask(mask, 10)
    assert torch.allclose(result, mask, atol=1e-5)


def test_sharp_step_smoothed(node):
    """A hard step edge should be smoothed after iterations."""
    mask = torch.zeros(1, 100, 100)
    mask[:, :, 50:] = 1.0
    (result,) = node.smooth_mask(mask, 10)
    # The boundary region should now have intermediate values.
    boundary_val = result[0, 50, 50].item()
    assert 0.1 < boundary_val < 0.9


def test_more_iterations_smoother(node):
    """More iterations should produce a smoother result."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 40:60, 40:60] = 1.0

    (result_5,) = node.smooth_mask(mask, 5)
    (result_20,) = node.smooth_mask(mask, 20)

    # Measure roughness via gradient magnitude.
    def roughness(t):
        dy = (t[:, 1:, :] - t[:, :-1, :]).abs().mean()
        dx = (t[:, :, 1:] - t[:, :, :-1]).abs().mean()
        return (dy + dx).item()

    assert roughness(result_20) < roughness(result_5)


def test_output_range(node):
    """Output should stay in [0, 1]."""
    mask = torch.rand(1, 64, 64)
    (result,) = node.smooth_mask(mask, 10)
    assert result.min().item() >= 0.0
    assert result.max().item() <= 1.0


def test_2d_input(node):
    """2D mask input should be normalised to 3D."""
    mask_2d = torch.ones(64, 64)
    (result,) = node.smooth_mask(mask_2d, 5)
    assert result.dim() == 3


def test_shape_preserved(node):
    """Output shape should match input shape."""
    mask = torch.zeros(2, 80, 120)
    (result,) = node.smooth_mask(mask, 5)
    assert result.shape == (2, 80, 120)


def test_already_smooth_gradient(node):
    """A smooth linear gradient should barely change."""
    mask = torch.linspace(0, 1, 100).unsqueeze(0).expand(1, 100, 100)
    mask = mask.clone()
    (result,) = node.smooth_mask(mask, 3)
    # Interior should be very close to original (edges may shift slightly).
    interior_diff = (result[0, 10:90, 10:90] - mask[0, 10:90, 10:90]).abs().max().item()
    assert interior_diff < 0.05


def test_only_increase_never_dims(node):
    """With only_increase=True, no pixel should decrease from the original."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 40:60, 40:60] = 1.0
    (result,) = node.smooth_mask(mask, 10, only_increase=True)
    # Every pixel should be >= original.
    assert (result >= mask - 1e-6).all()


def test_only_increase_fills_dips(node):
    """With only_increase=True, dips in the mask should get filled in."""
    mask = torch.ones(1, 100, 100)
    mask[:, 48:52, 48:52] = 0.0  # small hole
    (result,) = node.smooth_mask(mask, 20, only_increase=True)
    # The hole should be partially or fully filled.
    hole_val = result[0, 50, 50].item()
    assert hole_val > 0.5


def test_only_increase_vs_normal(node):
    """only_increase result should be >= normal smoothing everywhere."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 40:60, 40:60] = 1.0
    (normal,) = node.smooth_mask(mask, 10, only_increase=False)
    (increased,) = node.smooth_mask(mask, 10, only_increase=True)
    assert (increased >= normal - 1e-6).all()
