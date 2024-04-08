#!/usr/bin/env python3

from typing import List, Tuple
import rospy
import yaml
from std_msgs.msg import String

from geometry_msgs.msg import (
    Pose,
    Vector3,
)
from commander.msg import Goal
from commander.srv import (
    ExecuteTrajectory,
    PlanGoal,
    PlanSequence,
    PickPlace,
    GetTcpPose,
    VisualizePoses,
    SetEe,
)
from industrial_reconstruction_msgs.srv import (
    StartReconstruction,
    StartReconstructionRequest,
    StopReconstruction,
    StopReconstructionRequest,
)

from commander.utils import load_scene

CAPTURE = True

rospy.init_node("reconstruction")

ply_file_path = rospy.get_param("~ply_file_path", "/dev_ws/src/ur10e_examples/captures/raw.ply")

load_scene()

plan_goal_srv = rospy.ServiceProxy("commander/plan_goal", PlanGoal)
plan_sequence_srv = rospy.ServiceProxy("commander/plan_sequence", PlanSequence)
execute_trajectory_srv = rospy.ServiceProxy("commander/execute_trajectory", ExecuteTrajectory)
get_tcp_pose_srv = rospy.ServiceProxy("commander/get_tcp_pose", GetTcpPose)
set_ee_srv = rospy.ServiceProxy("commander/set_ee", SetEe)
pick_place_srv = rospy.ServiceProxy("commander/pick_place", PickPlace)

if CAPTURE:
    start_recon = rospy.ServiceProxy("/start_reconstruction", StartReconstruction)
    stop_recon = rospy.ServiceProxy("/stop_reconstruction", StopReconstruction)


def display_poses(poses: List[Pose], frame_id: str = "base_link") -> None:
    rospy.wait_for_service("/visualize_poses", timeout=10)
    visualize_poses = rospy.ServiceProxy("/visualize_poses", VisualizePoses)
    visualize_poses(frame_id, poses)


def gen_recon_msg(path: str) -> Tuple[StartReconstructionRequest, StopReconstructionRequest]:
    start_srv_req = StartReconstructionRequest()
    start_srv_req.tracking_frame = 'rgb_camera_tcp'
    start_srv_req.relative_frame = 'base_link'
    start_srv_req.translation_distance = 0.0
    start_srv_req.rotational_distance = 0.0
    start_srv_req.live = True
    start_srv_req.tsdf_params.voxel_length = 0.001
    start_srv_req.tsdf_params.sdf_trunc = 0.004
    start_srv_req.tsdf_params.min_box_values = Vector3(x=0.0, y=0.0, z=0.0)
    start_srv_req.tsdf_params.max_box_values = Vector3(x=0.0, y=0.0, z=0.0)
    start_srv_req.rgbd_params.depth_scale = 1000
    start_srv_req.rgbd_params.depth_trunc = 0.25
    start_srv_req.rgbd_params.convert_rgb_to_intensity = False

    stop_srv_req = StopReconstructionRequest()
    #path = path + datetime.now().strftime("%m_%d_%H_%M") + ".ply"
    stop_srv_req.mesh_filepath = path

    return start_srv_req, stop_srv_req

def execute_joint_states(joint_position):
    # Plan the goal using the stored joint position
    success = plan_goal_srv(Goal(joint_values=joint_position, vel_scale=0.02, acc_scale=0.02, planner='ptp')).success

    # Check if planning is successful
    if success:
        # Execute the trajectory
        success = execute_trajectory_srv()

        # Check if execution is successful
        if not success:
            rospy.loginfo("Failed to execute trajectory")
            exit()
    else:
        rospy.loginfo("Failed to plan")
        exit()

def callback(data):
    global triggered
    triggered = True  # Set triggered to True when something is published on the topic
    rospy.loginfo("Something has been published. Triggering the rest of the code.")

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('/label', String, callback)  # Use rospy.AnyMsg to subscribe to any message type
    rospy.spin()


triggered = True

#listener()

# This part will be executed after something is published on the topic
if triggered:

    cam_home = [-3.10042206640549, -2.131986729448063, 2.585211359700371, -3.5758736772415944, -1.5982155978220582, 0.0014838572949018819]
    plan_goal_srv(Goal(joint_values=cam_home, vel_scale=0.1, acc_scale=0.05, planner='ptp'))
    success = execute_trajectory_srv()

    success = set_ee_srv('rgb_camera_tcp')

    yaml_filename = "/dev_ws/src/ur10e_examples/config/joint_positions.yaml"

    try:
        with open(yaml_filename, 'r') as yaml_file:
            joint_positions_data = yaml.safe_load(yaml_file)
            joint_positions = joint_positions_data.get('joint_positions', [])
    except FileNotFoundError:
        rospy.loginfo(f"YAML file '{yaml_filename}' not found. Please run the code that saves joint positions first.")
        exit()

    if CAPTURE:
        start_recon_req, stop_recon_req = gen_recon_msg(ply_file_path)

    start = True

    for joint_position in joint_positions:
        
        execute_joint_states(joint_position)

        if start:
            print("Start recon")
            if CAPTURE:
                start_recon(start_recon_req)
                start = False


    if CAPTURE:
        stop_recon(stop_recon_req)