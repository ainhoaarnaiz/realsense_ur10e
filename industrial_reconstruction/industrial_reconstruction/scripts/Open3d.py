import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

# def read_mesh(path):
#     return o3d.io.read_triangle_mesh(path)


# def read_pcd(path):
#     return o3d.io.read_point_cloud(path)


# if __name__ == "__main__":
#     mesh = read_pcd("/home/justin/software2_files/ply_files/piece_mesh.obj")

#     o3d.visualization.draw_geometries([mesh])

#     # downpcd = mesh.voxel_down_sample(voxel_size=0.0005)

#     # o3d.visualization.draw_geometries([downpcd])

#     # create mesh from downsampled point cloud and visualize it


# # icp global registration
#     # source = mesh
#     # target = read_mesh("/home/justin/ply_files/path1.ply")
#     # threshold = 0.02

# load the digital file of the object from a .obj file
digital_mesh = o3d.io.read_triangle_mesh(
    "/home/justin/software2_files/ply_files/piece_mesh.obj")

# convert the mesh to a point cloud
digital_cloud = digital_mesh.sample_points_poisson_disk(number_of_points=50000)

# display the point cloud
o3d.visualization.draw_geometries([digital_cloud])
