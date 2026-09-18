import numpy as np
import pytest

from barista_cl.simulator import FIXED, GEOMETRY_ATOL, Simulator


class MatrixSim:
    def __init__(self, matrices):
        self.matrices = matrices

    def getObjectMatrix(self, handle, _relative_to):
        return self.matrices[handle]


def simulator_with_delta(delta):
    identity = np.eye(3, 4).reshape(-1).tolist()
    matrices = {name: identity.copy() for name in FIXED}
    matrices["wall"][11] += delta
    simulator = Simulator.__new__(Simulator)
    simulator.handles = {name: name for name in FIXED}
    simulator.sim = MatrixSim(matrices)
    simulator.fixed_geometry = {name: identity.copy() for name in FIXED}
    return simulator


def test_geometry_guard_allows_submillimetre_solver_settling():
    simulator_with_delta(0.000866).assert_geometry()


def test_geometry_guard_rejects_changes_beyond_tolerance():
    with pytest.raises(RuntimeError, match="wall.*max matrix delta"):
        simulator_with_delta(GEOMETRY_ATOL + 1e-4).assert_geometry()
