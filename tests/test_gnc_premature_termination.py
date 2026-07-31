"""GNC must anneal even when the initial estimate is already an LM fixed point.

Regression for the incremental-merge no-op bug: GTSAM's GncOptimizer checks
relativeCostTol from the very first GNC iteration. When the initial values
already sit at the LM optimum of the (weights~1) problem -- exactly what an
incremental merge feeds it -- the cost cannot change, the check fires at
iteration 0, and every factor keeps weight 1.0 regardless of how wrong it is.

A minimal synthetic pose graph (a handful of poses with one bad loop factor)
was tried extensively (single/double outliers, redundant loop closures,
offsets from 0.02 m to 100 m, tightened noise models, random search over
hundreds of configurations) and could not reproduce the iteration-0
`relativeCostTol` termination: GTSAM's mu-annealing schedule (GM) or
weight-binarization check (TLS) always wins first on graphs this small, so
`relativeCostTol` never becomes the binding stopping criterion. The bug
needs the scale/redundancy of an actual multi-session merge graph to
manifest, so the regression fixture below is a real GNC snapshot (gzip
committed to the repo, ~250 KB) rather than a synthetic construction. See
.superpowers/sdd/2026-07-31-gnc-noop-fix-and-connectivity-guard/
task-1-report.md for the full investigation.
"""
import gzip
import os
import shutil

import numpy as np
import pytest

pytest.importorskip("gtsam")
import gtsam

from utils.gtsam_pose_graph import PoseGraph

_FIXTURE_GZ = os.path.join(os.path.dirname(__file__), "fixtures", "step40_pose_graph.g2o.gz")

_REAL_DATASET_RESULTS_DIR = (
	"/Titan/dataset/data_opennavmap/map_multisession_eval/ucl_campus_aria/"
	"s00000_results_in_spgo_cc_seqmatch_master_gncgm_iqaigtd"
)


def _load_step40_graph(g2o_path: str):
	"""Read a g2o snapshot and add the anchor prior used by the merge pipeline."""
	graph, initial = gtsam.readG2o(g2o_path, True)
	k0 = min(initial.keys())
	graph.add(gtsam.PriorFactorPose3(
		k0, initial.atPose3(k0),
		gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3] * 6))))
	return graph, initial


@pytest.fixture
def step40_graph(tmp_path):
	g2o_path = tmp_path / "step40_pose_graph.g2o"
	with gzip.open(_FIXTURE_GZ, "rb") as fin, open(g2o_path, "wb") as fout:
		shutil.copyfileobj(fin, fout)
	return _load_step40_graph(str(g2o_path))


def test_gnc_gm_anneals_from_real_incremental_merge_fixture(step40_graph) -> None:
	"""GM must reject >=5 stale loop factors from the committed step-40 fixture.

	Diagnosed with relative_cost_tol=0.0: 5 of the incremental-merge loop
	factors are outliers (weight < 0.5). With GTSAM's default relativeCostTol
	(1e-5), GNC terminates at iteration 0 and returns all-ones weights
	(0 factors < 0.5) because the snapshot is already an LM fixed point.
	"""
	graph, initial = step40_graph
	_, weights = PoseGraph.optimize_pose_graph_with_GNC(graph, initial, loss='GM')
	assert (weights < 0.5).sum() >= 5, (
		f"only {(weights < 0.5).sum()} factors rejected: "
		"GNC terminated without annealing")


def test_gnc_tls_anneals_from_real_incremental_merge_fixture(step40_graph) -> None:
	"""TLS on the same fixture: verified to also reject >=5 stale loop factors.

	Unlike GM, TLS was NOT observed to be affected by the relativeCostTol
	premature-termination bug on this graph (its binary weight-convergence
	check wins first, before or after the fix). This asserts the actually
	observed behaviour: valid, binary (0/1) weights with the same 5 loop
	factors correctly rejected -- not a claim about the bug mechanism.
	"""
	graph, initial = step40_graph
	_, weights = PoseGraph.optimize_pose_graph_with_GNC(graph, initial, loss='TLS')
	assert weights.shape == (graph.size(),)
	assert np.isfinite(weights).all()
	assert np.all((weights >= 0.0) & (weights <= 1.0))
	assert (weights < 0.5).sum() >= 5, (
		f"only {(weights < 0.5).sum()} factors rejected on TLS fixture")


def _find_step40_g2o_on_disk() -> str:
	if not os.path.isdir(_REAL_DATASET_RESULTS_DIR):
		return ""
	for name in sorted(os.listdir(_REAL_DATASET_RESULTS_DIR)):
		if name.startswith("merge_") and name.endswith("_40"):
			g2o_path = os.path.join(_REAL_DATASET_RESULTS_DIR, name, "preds", "initial_pose_graph.g2o")
			if os.path.isfile(g2o_path):
				return g2o_path
	return ""


_DISK_G2O_PATH = _find_step40_g2o_on_disk()


@pytest.mark.skipif(
	not _DISK_G2O_PATH,
	reason=f"real dataset not mounted under {_REAL_DATASET_RESULTS_DIR} (extra cross-check only)",
)
def test_gnc_gm_anneals_from_mounted_dataset_snapshot() -> None:
	"""Extra cross-check against the live dataset path referenced in the design doc.

	Not required for the regression to be verified (see the fixture-based
	tests above, which run unconditionally); this just confirms the
	committed fixture still matches the live dataset when it is mounted.
	"""
	graph, initial = _load_step40_graph(_DISK_G2O_PATH)
	_, weights = PoseGraph.optimize_pose_graph_with_GNC(graph, initial, loss='GM')
	assert (weights < 0.5).sum() >= 5
