import cv2
import numpy as np
import os
import pyrealsense2 as rs

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

# Get the directory of the current Python script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the file path for saving the image
save_directory = current_directory  # You can change this to your desired directory
save_file_path = os.path.join(save_directory, "original.jpg")

# Wait for a coherent color frame
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()

if color_frame:
    # Convert color frame to numpy array
    frame = np.asanyarray(color_frame.get_data())


# Save the image with detected ArUco markers, the largest brown line, width length, and middle points
cv2.imwrite(save_file_path, frame)
print(f"Image saved at: {save_file_path}")