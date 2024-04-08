import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import copy

def find_middle_point(point_cloud):
    # Calculate the average y-value
    points = np.asarray(point_cloud.points)
    average_y = np.mean(points[:, 1])
    # Find the point closest to the average y-value
    distances = np.abs(points[:, 1] - average_y)
    closest_point_index = np.argmin(distances)
    point_cloud.colors[closest_point_index] = [1, 0, 0]
    closest_point_xyz = points[closest_point_index]

    return closest_point_xyz

def find_most_common_x(point_cloud, tolerance=0.0075):
    x_values = np.asarray(point_cloud.points)[:, 0]
    rounded_x_values = np.round(x_values / tolerance) * tolerance
    x_counts = Counter(rounded_x_values)
    most_common_x = max(x_counts, key=x_counts.get)
    return most_common_x

def filter_points_within_tolerance(point_cloud, target_x, tolerance=0.0075):
    x_values = np.asarray(point_cloud.points)[:, 0]
    filtered_indices = np.where(np.abs(x_values - target_x) <= tolerance)[0]
    filtered_points = point_cloud.select_by_index(filtered_indices)
    return filtered_points

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

def cluster_point_cloud(point_cloud):
    points = np.asarray(point_cloud.points)
    min_z_values = np.min(points[:, 2])
    max_z_values = np.max(points[:, 2])
    
    z_value = (min_z_values + max_z_values) / 2
    
    lower_bound = z_value - 0.0035
    upper_bound = z_value + 0.0055
    line_points = threshold_array(points, lower_bound, upper_bound)

    all_line_points = []
    for point in line_points:
        point[2] = z_value  # Set z-coordinate to cluster middle
    all_line_points.extend(line_points)

    line_cloud = create_point_cloud([all_line_points])
    common_x = find_most_common_x(line_cloud)
    filtered_points = filter_points_within_tolerance(line_cloud, common_x)
    middle_point = find_middle_point(filtered_points)

    return filtered_points, middle_point

def get_bounding_box_corners(point_cloud):
    points = np.asarray(point_cloud.points)
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    return min_bound, max_bound

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

# def crop_point_cloud(point_cloud, z_threshold):
#     points = np.asarray(point_cloud.points)
#     cropped_points = []
#     for point in points:
#         if point[2] >= z_threshold:
#             cropped_points.append(point)
#     if len(cropped_points) == 0:
#         print("Error: No points above the specified z-value threshold.")
#         return None
#     cropped_cloud = o3d.geometry.PointCloud()
#     cropped_cloud.points = o3d.utility.Vector3dVector(np.asarray(cropped_points))
#     return cropped_cloud

def create_point_cloud(arrays):
    point_cloud = o3d.geometry.PointCloud()
    for array in arrays:
        points = np.asarray(array)
        if len(points.shape) == 1:
            points = points.reshape(1, -1)  # Reshape to 2D if it's 1D
        point_cloud.points.extend(o3d.utility.Vector3dVector(points))
        colors = [[1.0, 1.0, 1.0] for _ in range(len(points))]  # Pink color
        point_cloud.colors.extend(o3d.utility.Vector3dVector(colors))
    return point_cloud

def main():
    file_path = "/dev_ws/src/ur10e_examples/captures/raw_A0.ply"
    point_cloud = load_point_cloud(file_path)
    n_clusters = 1

    min_bound, max_bound = get_bounding_box_corners(point_cloud)
    # min_bound -= np.array([-0.08, -0.08, -0.05]) #how to automatize this??
    # max_bound += np.array([-0.119, -0.08, -0.04]) #how to automatize this??
    min_bound -= np.array([-0.1, -0.08, -0.11]) #how to automatize this??
    max_bound += np.array([-0.02, -0.08, -0.07]) #how to automatize this??

    downsampled_cloud = downsample_point_cloud(point_cloud)
    cropped_cloud = crop_point_cloud(point_cloud, min_bound, max_bound)
    line_cloud, middle_point = cluster_point_cloud(copy.deepcopy(cropped_cloud))


    # Create a red point at the closest point to the average y-value
    red_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
    red_point.paint_uniform_color([1, 0, 0])  # Set color to red
    red_point.translate(middle_point)  # Translate to the closest point
    print(middle_point)

    # Visualize the combined line cloud with the bounding box
    #o3d.visualization.draw_geometries([line_cloud], window_name="Point Cloud with Bounding Box and Clustering")
    
    # Draw geometries
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud with Bounding Box and Clustering")
    vis.add_geometry(point_cloud)
    # vis.add_geometry(cropped_cloud)
    # vis.add_geometry(line_cloud)
    # vis.add_geometry(red_point)

    # Set background color to black
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])  # [R, G, B] values for black

    # Run visualization
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
