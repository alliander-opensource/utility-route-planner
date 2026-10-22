"""
Tests for the pipe ramming via weight.

Weight model (Layer 2 units: one step = one edge between adjacent centroids, length sqrt(3) * h):

    w_r = v_r * L / s + C_0          with s = sqrt(3) * h

    L    ramming length in metres (centroid to centroid)
    v_r  cost per metre of ramming, relative to ideal trenching (value 1)
    C_0  fixed cost of equipment and pits, in Layer 2 step units

Guarantee: a ramming must be cheaper than trenching between the same two pit cells whenever
that trench crosses closed paving ("gesloten verharding"). The worst case for the ramming is the
CHEAPEST trench, i.e. one that crosses a single cell of the cheapest closed paving and is otherwise
on the cheapest surface. For a trench of k steps, crossing n cells of value v_c, with all other cells
at v_side:

    trench = k * v_side + n * (v_c - v_side)

With v_r <= v_side and k >= L / s, the ramming is cheaper for every L, every k and every n >= 1 iff

    C_0 < v_c - v_min                                                          (*)

independent of ramming length and hexagon size. With v_c = 33 and v_min = 1, C_0 < 32.

Swap the local get_pipe_ramming_weight for the implementation in pipe_ramming.py once it exists.
"""

import heapq
import math

import pytest
from settings import Config

# ---------------------------------------------------------------------------------------------
# Parameters. Values for the benchmark preset (mcda_presets.py).
# ---------------------------------------------------------------------------------------------
V_MIN = 1.0  # Lower bound MCDA hexagon grid
V_CLOSED_PAVING_CHEAPEST = 33.0  # "gesloten verharding"
V_RAMMING_PER_M = 1.0  # cost per metre of ramming, relative to ideal trenching
C_SETUP = 30.0  # fixed cost per crossing, in Layer 2 step units; must satisfy (*)
HEXAGON_SIZE = 0.5
# MIN_RAMMING_LENGTH_M = 3.0
# MAX_RAMMING_LENGTH_M = 15.0


def get_pipe_ramming_weight(
    length_m: float,
    hexagon_size: float = Config.HEXAGON_SIZE,
    v_ramming_per_m: float = V_RAMMING_PER_M,
    c_setup: float = C_SETUP,
) -> float:
    """Weight of a pipe ramming via in Layer 2 step units: v_r * L / s + C_0."""
    step_length = math.sqrt(3) * hexagon_size
    return v_ramming_per_m * length_m / step_length + c_setup


def trench_cost(n_steps: int, n_closed: int, v_side: float, v_closed: float) -> float:
    """Cost of a straight trench of n_steps edges crossing n_closed closed-paving cells."""
    return n_steps * v_side + n_closed * (v_closed - v_side)


# ---------------------------------------------------------------------------------------------
# 1. The parameter condition itself.
# ---------------------------------------------------------------------------------------------
def test_setup_cost_satisfies_guarantee():
    assert V_RAMMING_PER_M <= V_MIN, "Guarantee (*) assumes ramming per metre is no costlier than the cheapest trench."
    assert C_SETUP < V_CLOSED_PAVING_CHEAPEST - V_MIN, (
        f"C_0 = {C_SETUP} breaks the guarantee; it must stay below v_c - v_min = {V_CLOSED_PAVING_CHEAPEST - V_MIN}."
    )


# ---------------------------------------------------------------------------------------------
# 2. Analytic worst case over every feasible straight crossing.
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("v_side", [1.0, 2.0, 3.0, 6.0, 10.0])
@pytest.mark.parametrize("n_closed", [1, 2, 5])
def test_ramming_beats_any_closed_paving_trench(v_side, n_closed):
    s = math.sqrt(3) * Config.HEXAGON_SIZE
    k_min = max(math.ceil(Config.MIN_PIPE_RAMMING_LENGTH_M / s), n_closed + 1)
    k_max = math.floor(Config.MAX_PIPE_RAMMING_LENGTH_M / s)
    for k in range(k_min, k_max + 1):
        length = k * s  # straight crossing along a grid axis: tightest case, as hex steps == L / s
        w_r = get_pipe_ramming_weight(length)
        w_t = trench_cost(k, n_closed, v_side, V_CLOSED_PAVING_CHEAPEST)
        assert w_r < w_t, f"L={length:.2f} m, n={n_closed}, v_side={v_side}: ramming {w_r:.2f} >= trench {w_t:.2f}"


# ---------------------------------------------------------------------------------------------
# 3. Brute force on an actual hexagonal grid: a one-cell-wide band of closed paving (the
#    narrowest possible in the grid), Dijkstra with mean-of-endpoints edge weights, and a
#    ramming between every pair of cells on opposite sides within the feasible length range.
# ---------------------------------------------------------------------------------------------
NEIGHBOURS = [(1, 0), (0, -1), (-1, 1), (1, -1), (-1, 0), (0, 1)]


def _centroid(q: int, r: int, h: float) -> tuple[float, float]:
    return 1.5 * h * q, math.sqrt(3) * h * (r + q / 2)


def _dijkstra(values: dict, source: tuple, target: tuple) -> float:
    dist = {source: 0.0}
    heap = [(0.0, source)]
    while heap:
        d, node = heapq.heappop(heap)
        if node == target:
            return d
        if d > dist.get(node, math.inf):
            continue
        for dq, dr in NEIGHBOURS:
            nb = (node[0] + dq, node[1] + dr)
            if nb not in values:
                continue
            nd = d + (values[node] + values[nb]) / 2
            if nd < dist.get(nb, math.inf):
                dist[nb] = nd
                heapq.heappush(heap, (nd, nb))
    return math.inf


@pytest.mark.parametrize("v_side", [1.0, 2.0, 3.0])
def test_ramming_beats_trench_on_hex_grid(v_side):
    h = HEXAGON_SIZE
    radius = 20
    # Band of closed paving at r == 0: every path from r < 0 to r > 0 must enter it at least once.
    values = {
        (q, r): (V_CLOSED_PAVING_CHEAPEST if r == 0 else v_side)
        for q in range(-radius, radius + 1)
        for r in range(-radius, radius + 1)
    }
    source = (0, -1)
    worst_margin = math.inf
    for q, r in values:
        if r <= 0:
            continue
        x0, y0 = _centroid(*source, h)
        x1, y1 = _centroid(q, r, h)
        length = math.hypot(x1 - x0, y1 - y0)
        if not MIN_RAMMING_LENGTH_M <= length <= MAX_RAMMING_LENGTH_M:
            continue
        w_t = _dijkstra(values, source, (q, r))
        w_r = get_pipe_ramming_weight(length, hexagon_size=h)
        worst_margin = min(worst_margin, w_t - w_r)
        assert w_r < w_t, f"target={(q, r)}, L={length:.2f}: ramming {w_r:.2f} >= trench {w_t:.2f}"
    assert worst_margin > 0


# ---------------------------------------------------------------------------------------------
# 4. Anti-zigzag: switching to a cheaper opposite side and back costs two crossings. Below the
#    critical stretch length the route must stay on its own side.
#
#    Gain of using the other side over D metres:  delta * D / s
#    Cost of crossing there and back:              2 * w_r(L)
#    Critical stretch:                             D_crit = 2 * w_r(L) * s / delta
# ---------------------------------------------------------------------------------------------
def critical_stretch_m(crossing_length_m: float, delta_value: float, hexagon_size: float = HEXAGON_SIZE) -> float:
    s = math.sqrt(3) * hexagon_size
    return 2 * get_pipe_ramming_weight(crossing_length_m, hexagon_size) * s / delta_value


def test_critical_stretch_pavement_versus_berm():
    # Pavement (3) on one side, berm (2) on the other: delta = 1 per step.
    d_crit = critical_stretch_m(crossing_length_m=10.0, delta_value=1.0)
    assert d_crit > 50.0, f"Side switching already pays off over {d_crit:.1f} m; raise C_0 or accept."
