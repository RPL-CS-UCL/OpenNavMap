import open3d as o3d
import numpy as np
import time

def draw_registration_result(source, target, transformation):
    source_temp = source.clone()
    target_temp = target.clone()

    source_temp.transform(transformation)

    # This is patched version for tutorial rendering.
    # Use `draw` function for you application.
    o3d.visualization.draw_geometries(
        [source_temp.to_legacy(),
         target_temp.to_legacy()],
        zoom=0.4459,
        front=[0.9288, -0.2951, -0.2242],
        lookat=[1.6784, 2.0612, 1.4451],
        up=[-0.3402, -0.9189, -0.1996])

demo_icp_pcds = o3d.data.DemoICPPointClouds()
source = o3d.t.io.read_point_cloud(demo_icp_pcds.paths[0])
target = o3d.t.io.read_point_cloud(demo_icp_pcds.paths[1])

# For Colored-ICP `colors` attribute must be of the same dtype as `positions` and `normals` attribute.
source.point["colors"] = source.point["colors"].to(
    o3d.core.Dtype.Float32) / 255.0
target.point["colors"] = target.point["colors"].to(
    o3d.core.Dtype.Float32) / 255.0

# Initial guess transform between the two point-cloud.
# ICP algortihm requires a good initial allignment to converge efficiently.
trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                         [-0.139, 0.967, -0.215, 0.7],
                         [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])

draw_registration_result(source, target, trans_init)

# Search distance for Nearest Neighbour Search [Hybrid-Search is used].
max_correspondence_distance = 0.07

# Initial alignment or source to target transform.
init_source_to_target = np.asarray([[0.862, 0.011, -0.507, 0.5],
                                    [-0.139, 0.967, -0.215, 0.7],
                                    [0.487, 0.255, 0.835, -1.4],
                                    [0.0, 0.0, 0.0, 1.0]])

# Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()

# Convergence-Criteria for Vanilla ICP
criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.000001,
                                       relative_rmse=0.000001,
                                       max_iteration=50)
# Down-sampling voxel-size.
voxel_size = 0.025

# Save iteration wise `fitness`, `inlier_rmse`, etc. to analyse and tune result.
save_loss_log = True

callback_after_iteration = lambda loss_log_map : print("Iteration Index: {}, Scale Index: {}, Scale Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
    loss_log_map["iteration_index"].item(),
    loss_log_map["scale_index"].item(),
    loss_log_map["scale_iteration_index"].item(),
    loss_log_map["fitness"].item(),
    loss_log_map["inlier_rmse"].item()))

s = time.time()
registration_icp = o3d.pipelines.registration.icp(source, target, max_correspondence_distance,
                            init_source_to_target, estimation, criteria,
                            voxel_size, callback_after_iteration)

icp_time = time.time() - s
print("Time taken by ICP: ", icp_time)
print("Inlier Fitness: ", registration_icp.fitness)
print("Inlier RMSE: ", registration_icp.inlier_rmse)

draw_registration_result(source, target, registration_icp.transformation)