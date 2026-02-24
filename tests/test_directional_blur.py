"""Tests for the DirectionalBlur node."""
import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from directional_blur import DirectionalBlur


@pytest.fixture
def node():
    return DirectionalBlur()


# ------------------------------------------------------------------
# Basic sanity
# ------------------------------------------------------------------

def test_radius_zero_unchanged(node):
    """radius=0 returns unchanged mask."""
    mask = torch.ones(1, 64, 64)
    (result,) = node.directional_blur(mask, 180.0, 0)
    assert result.shape == (1, 64, 64)
    assert torch.equal(result, mask)


def test_2d_input(node):
    """2D mask input is normalised to 3D output."""
    mask_2d = torch.ones(64, 64)
    (result,) = node.directional_blur(mask_2d, 180.0, 5)
    assert result.dim() == 3


# ------------------------------------------------------------------
# Directional behaviour — bbox mode
# ------------------------------------------------------------------

def test_bbox_blur_downward(node):
    """angle=180 (down): bottom edge gets more blur than top edge."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 30:70, 30:70] = 1.0
    (result,) = node.directional_blur(mask, 180.0, 10, "bbox")
    bottom = result[0, 75, 50].item()
    top = result[0, 25, 50].item()
    assert bottom > top


def test_bbox_blur_upward(node):
    """angle=0 (up): top edge gets more blur than bottom edge."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 30:70, 30:70] = 1.0
    (result,) = node.directional_blur(mask, 0.0, 10, "bbox")
    top = result[0, 25, 50].item()
    bottom = result[0, 75, 50].item()
    assert top > bottom


def test_bbox_blur_rightward(node):
    """angle=90 (right): right edge gets more blur than left edge."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 30:70, 30:70] = 1.0
    (result,) = node.directional_blur(mask, 90.0, 10, "bbox")
    right = result[0, 50, 75].item()
    left = result[0, 50, 25].item()
    assert right > left


def test_bbox_trailing_edge_unchanged(node):
    """The trailing edge (opposite to angle) stays close to original."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 30:70, 30:70] = 1.0
    (result,) = node.directional_blur(mask, 180.0, 10, "bbox")
    # Row 25 is above the mask — trailing edge, should stay near 0.
    top_above = result[0, 25, 50].item()
    assert top_above < 0.05


def test_bbox_interior_gradient(node):
    """Interior blends smoothly — leading side dimmer, trailing side fuller."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 30:70, 30:70] = 1.0
    (result,) = node.directional_blur(mask, 180.0, 10, "bbox")
    top_interior = result[0, 35, 50].item()  # trailing side of interior
    bottom_interior = result[0, 65, 50].item()  # leading side of interior
    # Trailing side should be closer to original (1.0).
    assert top_interior > bottom_interior


# ------------------------------------------------------------------
# Directional behaviour — sdf mode
# ------------------------------------------------------------------

def test_sdf_blur_downward(node):
    """SDF mode: angle=180 (down): bottom edge gets more blur than top."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 30:70, 30:70] = 1.0
    (result,) = node.directional_blur(mask, 180.0, 10, "sdf")
    bottom = result[0, 75, 50].item()
    top = result[0, 25, 50].item()
    assert bottom > top


def test_sdf_blur_upward(node):
    """SDF mode: angle=0 (up): top edge gets more blur than bottom."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 30:70, 30:70] = 1.0
    (result,) = node.directional_blur(mask, 0.0, 10, "sdf")
    top = result[0, 25, 50].item()
    bottom = result[0, 75, 50].item()
    assert top > bottom


def test_sdf_trailing_edge_unchanged(node):
    """SDF: trailing edge stays close to original."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 30:70, 30:70] = 1.0
    (result,) = node.directional_blur(mask, 180.0, 10, "sdf")
    top_above = result[0, 25, 50].item()
    assert top_above < 0.05


# ------------------------------------------------------------------
# Multi-pass
# ------------------------------------------------------------------

def test_multipass_runs(node):
    """Large radius triggers multi-pass without error."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 30:70, 30:70] = 1.0
    (result,) = node.directional_blur(mask, 180.0, 30, "bbox")
    assert result.shape == (1, 100, 100)


def test_multipass_single_pass_flag(node):
    """single_pass=True uses one big kernel."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 30:70, 30:70] = 1.0
    (result,) = node.directional_blur(mask, 180.0, 20, "bbox", True)
    assert result.shape == (1, 100, 100)


# ------------------------------------------------------------------
# No brightening
# ------------------------------------------------------------------

def test_no_brightening(node):
    """Result should never exceed the original mask value."""
    mask = torch.zeros(1, 100, 100)
    mask[:, 30:70, 30:70] = 1.0
    (result,) = node.directional_blur(mask, 180.0, 10, "bbox")
    assert result.max().item() <= 1.001
