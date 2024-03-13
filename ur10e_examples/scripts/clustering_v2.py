import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans

def load_point_cloud(file_path):
    return o3d.io.read_point_cloud(file_path)

def downsample_point_cloud(point_cloud, voxel_size=0.002):
    return point_cloud.voxel_down_sample(voxel_size=voxel_size)

def threshold_array(array, lower_bound, upper_bound):
    within_threshold = []
    for element in array:
        if lower_bound <= element[2] <= upper_bound:
            within_threshold.append(element)
    return within_threshold

def cluster_point_cloud(point_cloud, n_clusters=3):
    points = np.asarray(point_cloud.points)
    if len(points) == 0:
        print("Error: Point cloud is empty.")
        return None, None, None
    
    z_values = points[:, 2].reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(z_values)
    labels = kmeans.labels_
    colors = np.random.rand(n_clusters, 3)
    point_cloud.colors = o3d.utility.Vector3dVector(colors[labels])
    
    min_z_values = []
    max_z_values = []
    for i in range(n_clusters):
        cluster_points = points[labels == i]
        min_z_values.append(np.min(cluster_points[:, 2]))
        max_z_values.append(np.max(cluster_points[:, 2]))
    
    cluster_middles = []
    for i in range(len(min_z_values)):
        avg = (min_z_values[i] + max_z_values[i]) / 2
        cluster_middles.append(avg)

    all_line_points = []
    for i in range(n_clusters):
        z_value = cluster_middles[i]
        cluster_points = points[labels == i]
        lower_bound = z_value - 0.0035
        upper_bound = z_value + 0.0055
        line_points = threshold_array(cluster_points, lower_bound, upper_bound)
        for point in line_points:
            point[2] = z_value  # Set z-coordinate to cluster middle
        all_line_points.extend(line_points)

    line_cloud = create_point_cloud([all_line_points])

    return point_cloud, line_cloud

def get_bounding_box_corners(point_cloud):
    points = np.asarray(point_cloud.points)
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    return min_bound, max_bound

def create_bounding_box(min_bound, max_bound):
    corners = np.array([[min_bound[0], min_bound[1], min_bound[2]],
                        [max_bound[0], min_bound[1], min_bound[2]],
                        [max_bound[0], max_bound[1], min_bound[2]],
                        [min_bound[0], max_bound[1], min_bound[2]],
                        [min_bound[0], min_bound[1], max_bound[2]],
                        [max_bound[0], min_bound[1], max_bound[2]],
                        [max_bound[0], max_bound[1], max_bound[2]],
                        [min_bound[0], max_bound[1], max_bound[2]]])
    edges = [[0, 1], [1, 2], [2, 3], [3, 0],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [[1, 0, 0] for i in range(len(edges))]
    bounding_box = o3d.geometry.LineSet()
    bounding_box.points = o3d.utility.Vector3dVector(corners)
    bounding_box.lines = o3d.utility.Vector2iVector(edges)
    bounding_box.colors = o3d.utility.Vector3dVector(colors)
    return bounding_box

def crop_point_cloud(point_cloud, min_bound, max_bound):
    points = np.asarray(point_cloud.points)
    cropped_points = []
    for point in points:
        if point[0] <= min_bound[0] and  point[1] >= min_bound[1] and  point[2] >= min_bound[2]:
            if point[0] >= max_bound[0] and  point[1] <= max_bound[1] and  point[2] <= max_bound[2]:
                cropped_points.append(point)
    if len(cropped_points) == 0:
        print("Error: No points within the bounding box.")
        return None
    cropped_cloud = o3d.geometry.PointCloud()
    cropped_cloud.points = o3d.utility.Vector3dVector(np.asarray(cropped_points))
    return cropped_cloud

def create_point_cloud(arrays):
    point_cloud = o3d.geometry.PointCloud()
    for array in arrays:
        points = np.asarray(array)
        if len(points.shape) == 1:
            points = points.reshape(1, -1)  # Reshape to 2D if it's 1D
        point_cloud.points.extend(o3d.utility.Vector3dVector(points))
        colors = [[1, 0, 1] for _ in range(len(points))]  # Pink color
        point_cloud.colors.extend(o3d.utility.Vector3dVector(colors))
    return point_cloud

def main():
    file_path = "/dev_ws/src/ur10e_examples/captures/A5_v4.ply"
    point_cloud = load_point_cloud(file_path)
    n_clusters = 4

    if point_cloud is None:
        return

    min_bound, max_bound = get_bounding_box_corners(point_cloud)
    min_bound -= np.array([-0.08, -0.08, -0.05]) #how to automatize this??
    max_bound += np.array([-0.119, -0.08, -0.04]) #how to automatize this??
    bounding_box = create_bounding_box(min_bound, max_bound)

    downsampled_cloud = downsample_point_cloud(point_cloud)
    if downsampled_cloud is None:
        return
    
    cropped_cloud = crop_point_cloud(point_cloud, min_bound, max_bound)
    if cropped_cloud is None:
        return    

    cluster_cloud, line_cloud = cluster_point_cloud(cropped_cloud, n_clusters)
    if line_cloud is None:
        return
    
    # Visualize the combined line cloud with the bounding box
    #o3d.visualization.draw_geometries([line_cloud, bounding_box], window_name="Point Cloud with Bounding Box and Clustering")
    o3d.visualization.draw_geometries([cluster_cloud, line_cloud, bounding_box], window_name="Point Cloud with Bounding Box and Clustering")

if __name__ == "__main__":
    main()
