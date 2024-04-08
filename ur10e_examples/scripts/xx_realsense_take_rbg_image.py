import cv2
import numpy as np
import pyrealsense2 as rs

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

save_file_path = "/dev_ws/src/ur10e_examples/captures/original.jpg"

# Wait for a coherent color frame
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()

if color_frame:
    # Convert color frame to numpy array
    frame = np.asanyarray(color_frame.get_data())


# Save the image with detected ArUco markers, the largest brown line, width length, and middle points
cv2.imwrite(save_file_path, frame)
print(f"Image saved at: {save_file_path}")