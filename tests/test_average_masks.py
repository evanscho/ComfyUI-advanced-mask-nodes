"""Smoke tests for the AverageOverlappingMasks node."""
import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from average_masks import AverageOverlappingMasks


@pytest.fixture
def node():
    return AverageOverlappingMasks()


def test_full_overlap_identical(node):
    """Both masks identical → output equals input."""
    mask = torch.ones(1, 64, 64) * 0.8
    (result,) = node.average_masks(mask, mask)
    assert result.shape == (1, 64, 64)
    assert torch.allclose(result, mask)


def test_no_overlap_disjoint(node):
    """Disjoint regions keep their values."""
    a = torch.zeros(1, 100, 100)
    b = torch.zeros(1, 100, 100)
    a[:, :50, :] = 0.6
    b[:, 50:, :] = 0.4
    (result,) = node.average_masks(a, b)
    assert torch.allclose(result[:, :50, :], torch.full((1, 50, 100), 0.6))
    assert torch.allclose(result[:, 50:, :], torch.full((1, 50, 100), 0.4))


def test_partial_overlap(node):
    """Averaged where both non-zero, passthrough elsewhere."""
    a = torch.zeros(1, 100, 100)
    b = torch.zeros(1, 100, 100)
    a[:, 20:80, :] = 0.8
    b[:, 50:100, :] = 0.6
    (result,) = node.average_masks(a, b)
    assert torch.allclose(result[:, 20:50, :], torch.full((1, 30, 100), 0.8))
    assert torch.allclose(result[:, 50:80, :], torch.full((1, 30, 100), (0.8 + 0.6) / 2.0))
    assert torch.allclose(result[:, 80:100, :], torch.full((1, 20, 100), 0.6))
    assert (result[:, :20, :] == 0).all()


def test_one_mask_zero(node):
    """Zero + non-zero → output equals non-zero mask."""
    a = torch.zeros(1, 64, 64)
    b = torch.ones(1, 64, 64) * 0.5
    (result,) = node.average_masks(a, b)
    assert torch.allclose(result, b)


def test_both_zero(node):
    """Both zero → all-zero output."""
    a = torch.zeros(1, 64, 64)
    b = torch.zeros(1, 64, 64)
    (result,) = node.average_masks(a, b)
    assert (result == 0).all()


def test_2d_inputs(node):
    """2D mask inputs are normalised to 3D."""
    (result,) = node.average_masks(torch.ones(64, 64) * 0.4, torch.ones(64, 64) * 0.6)
    assert result.dim() == 3
    assert torch.allclose(result, torch.full((1, 64, 64), 0.5))


def test_mismatched_spatial_dims(node):
    """Mismatched spatial dimensions raise ValueError."""
    with pytest.raises(ValueError):
        node.average_masks(torch.zeros(1, 64, 64), torch.zeros(1, 32, 32))


def test_batch_broadcasting(node):
    """Batch dim 1 broadcasts to match batch dim N."""
    (result,) = node.average_masks(torch.ones(1, 64, 64) * 0.4, torch.ones(3, 64, 64) * 0.6)
    assert result.shape == (3, 64, 64)
    assert torch.allclose(result, torch.full((3, 64, 64), 0.5))


def test_presence_threshold_filters_noise(node):
    """Values below threshold are treated as absent."""
    a = torch.zeros(1, 64, 64)
    b = torch.zeros(1, 64, 64)
    a[:, :, :] = 0.005  # below default threshold of 0.01
    b[:, :, :] = 0.8
    (result,) = node.average_masks(a, b, presence_threshold=0.01)
    # a is below threshold → treated as absent, result should equal b
    assert torch.allclose(result, b)


def test_presence_threshold_zero_strict(node):
    """threshold=0 treats any non-zero value as present (original behaviour)."""
    a = torch.zeros(1, 64, 64)
    b = torch.zeros(1, 64, 64)
    a[:, :, :] = 0.005
    b[:, :, :] = 0.8
    (result,) = node.average_masks(a, b, presence_threshold=0.0, blend_threshold=0.0)
    # Both present → averaged
    expected = (0.005 + 0.8) / 2.0
    assert torch.allclose(result, torch.full((1, 64, 64), expected))
