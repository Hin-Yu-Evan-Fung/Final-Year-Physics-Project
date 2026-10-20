import pytest
import numpy as np
import hsmc
import time

@pytest.fixture
def initialized_engine():
    """Fixture to quickly spawn a fresh engine state for testing."""
    params = hsmc.HardSphereParams(0.5) # Example radius/interaction parameter
    engine = hsmc.HardSphereEngine(L=10.0, params=params)
    
    # Initialize engine with some starting mock positions (e.g., 3 particles)
    # Ensure this matches your specific set_particles signature or layout
    init_pos = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0],
        [4.0, 4.0, 4.0]
    ], dtype=np.float64)
    
    engine.set_particles(init_pos)
    return engine


def test_engine_initialization(initialized_engine):
    """Verify size constraints register accurately."""
    assert initialized_engine.size() == 3


def test_successful_batch_run(initialized_engine):
    """Verify that a valid array of moves executes completely."""
    indices = np.array([0, 1], dtype=np.uint32)
    
    # 2 rows, exactly 3 columns (matching HardSphere Move DOF)
    moves = np.array([
        [0.1, 0.0, 0.0],
        [0.0, -0.1, 0.0]
    ], dtype=np.float64)
    
    # If try_run_batch returns accepted count, track it
    accepted = initialized_engine.try_run_batch(indices, moves)
    assert isinstance(accepted, int)
    assert accepted <= len(indices)


def test_noconvert_type_checking_fail(initialized_engine):
    """Verify nanobind's .noconvert() blocks mismatched dtypes instantly."""
    indices = np.array([0, 1], dtype=np.uint32)
    
    # Malicious type float32 instead of expected float64
    bad_type_moves = np.array([
        [0.1, 0.0, 0.0],
        [0.0, -0.1, 0.0]
    ], dtype=np.float32)

    with pytest.raises(TypeError, match="incompatible function arguments"):
        initialized_engine.try_run_batch(indices, bad_type_moves)


def test_noconvert_shape_checking_fail(initialized_engine):
    """Verify nanobind blocks incorrect Degrees of Freedom matrix dimensions."""
    indices = np.array([0, 1], dtype=np.uint32)
    
    # 4 columns instead of the hardcoded 3 DOFs
    bad_shape_moves = np.array([
        [0.1, 0.0, 0.0, 9.9],
        [0.0, -0.1, 0.0, 9.9]
    ], dtype=np.float64)

    with pytest.raises(TypeError, match="incompatible function arguments"):
        initialized_engine.try_run_batch(indices, bad_shape_moves)


def test_non_contiguous_array_rejection(initialized_engine):
    """Verify that sliced/non-contiguous matrices are cleanly rejected."""
    indices = np.array([0, 1], dtype=np.uint32)
    
    # Make a larger matrix and slice it to create non-contiguous stride steps
    large_moves = np.random.rand(10, 3).astype(np.float64)
    sliced_moves = large_moves[::2, :] # Non-contiguous view

    with pytest.raises(TypeError):
        initialized_engine.try_run_batch(indices, sliced_moves)

