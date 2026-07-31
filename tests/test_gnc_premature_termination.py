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
`relativeCostTol` never becomes the binding stopping criterion. The bug is
real but needs the scale/redundancy of an actual multi-session merge graph
to manifest -- confirmed below against a real GNC snapshot from the
map-merging benchmark (see docs/.../task-1-report.md for the investigation).
"""
import os

import numpy as np
import pytest

pytest.importorskip("gtsam")
import gtsam

from utils.gtsam_pose_graph import PoseGraph

_RESULTS_DIR = (
	"/Titan/dataset/data_opennavmap/map_multisession_eval/ucl_campus_aria/"
	"s00000_results_in_spgo_cc_seqmatch_master_gncgm_iqaigtd"
)


def _find_step40_g2o() -> str:
	if not os.path.isdir(_RESULTS_DIR):
		return ""
	for name in sorted(os.listdir(_RESULTS_DIR)):
		if name.startswith("merge_") and name.endswith("_40"):
			g2o_path = os.path.join(_RESULTS_DIR, name, "preds", "initial_pose_graph.g2o")
			if os.path.isfile(g2o_path):
				return g2o_path
	return ""


_G2O_PATH = _find_step40_g2o()


@pytest.mark.skipif(
	not _G2O_PATH,
	reason=f"real GNC snapshot not available under {_RESULTS_DIR}",
)
def test_gnc_anneals_from_real_incremental_merge_snapshot() -> None:
	"""Step-40 g2o snapshot: GNC(GM) must reject >=5 stale loop factors.

	Diagnosed with relative_cost_tol=0.0: 5 of the incremental-merge loop
	factors are outliers (weight < 0.5). With GTSAM's default relativeCostTol
	(1e-5), GNC terminates at iteration 0 and returns all-ones weights
	(0 factors < 0.5) because the snapshot is already an LM fixed point.
	"""
	graph, initial = gtsam.readG2o(_G2O_PATH, True)
	k0 = min(initial.keys())
	graph.add(gtsam.PriorFactorPose3(
		k0, initial.atPose3(k0),
		gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3] * 6))))

	_, weights = PoseGraph.optimize_pose_graph_with_GNC(graph, initial, loss='GM')
	assert (weights < 0.5).sum() >= 5, (
		f"only {(weights < 0.5).sum()} factors rejected: "
		"GNC terminated without annealing")
