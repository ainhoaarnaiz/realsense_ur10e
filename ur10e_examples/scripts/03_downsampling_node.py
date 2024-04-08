#!/usr/bin/env python3

import rospy
import open3d as o3d
import numpy as np
from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud, ChannelFloat32
from collections import Counter
from geometry_msgs.msg import Point
import os
import time

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

def origin(point_cloud):
    min_bound, max_bound = get_bounding_box_corners(point_cloud)
    min_bound -= np.array([-0.1, -0.08, -0.11]) #how to automatize this??
    max_bound += np.array([-0.02, -0.08, -0.07]) #how to automatize this??
    cropped_cloud = crop_point_cloud(point_cloud, min_bound, max_bound)
    line_cloud, mid_point = cluster_point_cloud(cropped_cloud)

    middle_point = Point()
    middle_point.x = mid_point[0]
    middle_point.y = mid_point[1]
    middle_point.z = mid_point[2]

    # Publish the Point message
    pub2 = rospy.Publisher('/middle_point', Point, queue_size=10)
    pub2.publish(middle_point)
    rospy.loginfo("Published middle point: %s", middle_point)


def down_point_cloud(file_path, save_path, voxel_size=0.002):

    

    # Load your point cloud
    point_cloud = o3d.io.read_point_cloud(file_path)
    downsampled_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)

    origin = point_cloud.get_center()

    # o3d.io.write_point_cloud(save_path, downsampled_cloud)

    # Convert Open3D PointCloud to sensor_msgs/PointCloud
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors) * 255.0  # Assuming colors are in the range [0, 255]

    cloud_msg = PointCloud()
    cloud_msg.header.stamp = rospy.Time.now()
    cloud_msg.header.frame_id = "base_link"  # Change this to your desired frame_id

    for i in range(points.shape[0]):
        point = [points[i, 0] + origin[0], points[i, 1] + origin[1], points[i, 2] + origin[2]]

        # Create a Point32 message for each point
        point_msg = Point32()
        point_msg.x, point_msg.y, point_msg.z = point

        # Add point coordinates to the message
        cloud_msg.points.append(point_msg)

        # Add color data to the message
        color_channel = ChannelFloat32()
        # Assuming colors are in the range [0, 255]
        color_channel.values.append(colors[i, 0])
        color_channel.values.append(colors[i, 1])
        color_channel.values.append(colors[i, 2])
        cloud_msg.channels.append(color_channel)

    # Publish the cropped point cloud
    pub = rospy.Publisher('/downsampled_pc', PointCloud, queue_size=10)
    pub.publish(cloud_msg)

    return downsampled_cloud

def wait_for_file(file_path, timeout=60):
    """
    Wait until the file exists or until timeout (in seconds) is reached.
    """
    start_time = time.time()
    while not os.path.exists(file_path):
        time.sleep(1)


if __name__ == '__main__':
    rospy.init_node('point_cloud_cropper_node')
    ply_file_path = rospy.get_param("~ply_file_path", "/dev_ws/src/ur10e_examples/captures/raw.ply")
    ply_save_path = rospy.get_param("~ply_save_path", "/dev_ws/src/ur10e_examples/captures/cropped.ply")

    try:
        while not rospy.is_shutdown():
            wait_for_file(ply_file_path)
            down_sample = down_point_cloud(ply_file_path, ply_save_path)
            origin(down_sample)
            print("hola")
            rospy.loginfo("Processed point cloud.")
            rate = rospy.Rate(0.1)  # 0.1 Hz (every 10 seconds)
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
