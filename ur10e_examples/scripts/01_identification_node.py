#!/usr/bin/env python3

import rospy
from keras.models import load_model
import numpy as np
import pyrealsense2 as rs
import cv2
from commander.msg import Goal
from commander.srv import (
    ExecuteTrajectory,
    PlanGoal,
    PlanSequence,
    PickPlace,
    GetTcpPose,
    SetEe,
)
from std_msgs.msg import String

from commander.utils import load_scene

rospy.init_node("tile_identification")

load_scene()

plan_goal_srv = rospy.ServiceProxy("commander/plan_goal", PlanGoal)
plan_sequence_srv = rospy.ServiceProxy("commander/plan_sequence", PlanSequence)
execute_trajectory_srv = rospy.ServiceProxy("commander/execute_trajectory", ExecuteTrajectory)
get_tcp_pose_srv = rospy.ServiceProxy("commander/get_tcp_pose", GetTcpPose)
set_ee_srv = rospy.ServiceProxy("commander/set_ee", SetEe)
pick_place_srv = rospy.ServiceProxy("commander/pick_place", PickPlace)

# Initialize the publisher for /label topic
label_pub = rospy.Publisher('/label', String, queue_size=1)

cam_home = [-4.062083546315328, -1.150475339298584, 1.7826998869525355, -3.772630830804342, 0.012788206338882446, -7.3734913961232e-05]
plan_goal_srv(Goal(joint_values=cam_home, vel_scale=0.1, acc_scale=0.05, planner='ptp'))
success = execute_trajectory_srv()

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

# Wait for a coherent color frame
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()

if color_frame:
    model_path = "/dev_ws/src/ur10e_examples/scripts/keras_model.h5"
    label_path = "/dev_ws/src/ur10e_examples/scripts/labels.txt"

    # Load the model
    model = load_model(model_path, compile=False)

    # Load the labels
    with open(label_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # Convert the color frame to a numpy array
    image = np.asanyarray(color_frame.get_data())

    # Resize the image to the model's input shape (224x224)
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Reshape and normalize the image
    image_input = image_resized.astype(np.float32) / 255.0  # Normalize to [0,1]
    image_input = np.expand_dims(image_input, axis=0)  # Add batch dimension

    # Predict using the model
    prediction = model.predict(image_input)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    
    # Publish the class name into the ROS topic /label
    label_pub.publish(class_name[2:])  # Publish class_name[2:] to /label topic

    # Display the image with prediction
    # cv2.putText(image, f"Class: {class_name[2:]}, Confidence: {confidence_score*100:.2f}%", 
    #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # cv2.imshow("RealSense Image", image)
    # cv2.waitKey(0)

# Release the RealSense pipeline
pipeline.stop()
cv2.destroyAllWindows()

cam_home = [-3.10042206640549, -2.131986729448063, 2.585211359700371, -3.5758736772415944, -1.5982155978220582, 0.0014838572949018819]
plan_goal_srv(Goal(joint_values=cam_home, vel_scale=0.1, acc_scale=0.05, planner='ptp'))
success = execute_trajectory_srv()
