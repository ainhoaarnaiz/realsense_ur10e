import numpy as np
import open3d as o3d

# Load the digital file of the object from a .ply file
digital_mesh = o3d.io.read_triangle_mesh(
    "/home/justin/software2_files/digital.ply")

# Convert the mesh to a point cloud
digital_cloud = digital_mesh.sample_points_poisson_disk(number_of_points=50000)

# # downsample the digital point cloud
# digital_cloud = digital_cloud.voxel_down_sample(voxel_size=0.1)

# # # display the downsampled digital point cloud
# # o3d.visualization.draw_geometries([digital_cloud])

# Load the mesh created from the scan of the physical object
scan_mesh = o3d.io.read_triangle_mesh(
    "/home/justin/software2_files/ply_files/path2.ply")

# Convert the mesh to a point cloud
scan_cloud = scan_mesh.sample_points_poisson_disk(number_of_points=50000)

# downsample the scan point cloud
scan_cloud = scan_cloud.voxel_down_sample(voxel_size=0.001)

# scale the scan point cloud
scan_cloud = scan_cloud.scale(1500, center=scan_cloud.get_center())

# display the downsampled scan point cloud
o3d.visualization.draw_geometries([digital_cloud, scan_cloud])


# # Load the two point clouds
# source_cloud = digital_cloud
# target_cloud = scan_cloud

# # Create a global registration object
# global_registration = o3d.registration.RegistrationResult()

# # Perform global registration
# global_registration = o3d.registration.registration_icp(
#     source_cloud, target_cloud, 0.05)

# # Print the transformation matrix
# print(global_registration.transformation)

# # Apply the transformation to the source cloud
# source_cloud.transform(global_registration.transformation)

# # Draw the aligned point clouds
# o3d.visualization.draw_geometries([source_cloud, target_cloud])


# # display the scan point cloud
# o3d.visualization.draw_geometries([scan_cloud])

# # create an ICP registration object and set the parameters
# icp_reg = o3d.registration.registration_icp(
#     digital_cloud, scan_cloud, 0.02, np.identity(4),
#     o3d.registration.TransformationEstimationPointToPoint())

# # align the digital point cloud to the scan point cloud
# icp_registration.execute()

# source = digital_cloud
# target = scan_cloud
# threshold = 0.02
# # trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
# #                          [-0.139, 0.967, -0.215, 0.7],
# #                          [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]]


# def draw_registration_result(source, target, transformation):
#     source_temp = copy.deepcopy(source)
#     target_temp = copy.deepcopy(target)
#     source_temp.paint_uniform_color([1, 0.706, 0])
#     target_temp.paint_uniform_color([0, 0.651, 0.929])
#     source_temp.transform(transformation)
#     o3d.visualization.draw_geometries([source_temp, target_temp],
#                                       zoom=0.4459,
#                                       front=[0.9288, -0.2951, -0.2242],
#                                       lookat=[1.6784, 2.0612, 1.4451],
#                                       up=[-0.3402, -0.9189, -0.1996])
