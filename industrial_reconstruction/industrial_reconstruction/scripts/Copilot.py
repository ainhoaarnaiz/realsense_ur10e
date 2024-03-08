import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def read_mesh(path):
    return o3d.io.read_triangle_mesh(path)

def read_pcd(path):
    return o3d.io.read_point_cloud(path)

