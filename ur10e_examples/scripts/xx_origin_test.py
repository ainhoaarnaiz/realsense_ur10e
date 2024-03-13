import rospy
from geometry_msgs.msg import Pose, Point, Quaternion
from moveit_msgs.msg import CollisionObject, PlanningScene, ObjectColor
from shape_msgs.msg import Mesh, MeshTriangle
from std_msgs.msg import ColorRGBA
import open3d as o3d
import numpy as np
from moveit_commander import PlanningSceneInterface


rospy.init_node("reconstruction")

def create_mesh(cloud, center):
    # Downsample the point cloud
    #downsampled_point_cloud = cloud.voxel_down_sample(voxel_size=0.01)

    # Apply normal estimation
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Surface reconstruction using Poisson reconstruction
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=9)
    
    # Convert Open3D TriangleMesh to ROS Mesh message
    mesh_msg = Mesh()
    mesh_msg.vertices = [Point(x=float(pt[0] - center[0]), y=float(pt[1] - center[1]), z=float(pt[2] - center[2])) for pt in mesh[0].vertices]

    # Create MeshTriangle messages for each triangle
    mesh_msg.triangles = [MeshTriangle(vertex_indices=[int(idx) for idx in face]) for face in mesh[0].triangles]

    return mesh_msg

# Load point cloud from a ply file
ply_file_path = '/dev_ws/src/ur10e_examples/captures/raw_02.ply'
cloud = o3d.io.read_point_cloud(ply_file_path)
all_points = np.asarray(cloud.points)
center = np.mean(all_points, axis=0)

mesh = create_mesh(cloud, center)

scene_interface = PlanningSceneInterface()
scene = PlanningScene()
scene.is_diff = True

# Define the desired position of the mesh (relative to the center of the point cloud)
# move_x = -0.7
# move_z = 0.5

# Define the position of the mesh in the world frame
mesh_position = Point(x=center[0], y=center[1], z=center[2])

# Define the orientation of the mesh (assuming no rotation)
mesh_orientation = Quaternion(0.0, 0.0, 0.0, 1.0)

# Create the CollisionObject
co = CollisionObject()
co.header.frame_id = 'base_link'
co.header.stamp = rospy.Time.now()
co.mesh_poses = [Pose(position=mesh_position, orientation=mesh_orientation)]
co.id = 'chair'
co.operation = CollisionObject.ADD
co.meshes.append(mesh)
scene.world.collision_objects.append(co)

# Add color to the object
oc = ObjectColor()
oc.id = 'chair'
oc.color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.7)
scene.object_colors.append(oc)

# Apply the planning scene
scene_interface.apply_planning_scene(scene)

rospy.sleep(1)  # Wait for the scene to be updated

print("done")
