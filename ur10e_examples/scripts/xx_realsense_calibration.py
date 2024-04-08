import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import os

# Get the directory of the current Python script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the file path
file_path = "/dev_ws/src/ur10e_examples/calibration/realsense_calibration_data.npz"

# Initialize the Realsense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

# Prepare the calibration pattern (chessboard)
chessboard_size = (8, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((chessboard_size[1]*chessboard_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Number of images to capture
num_images_required = 20
count_images = 0

print("Starting Realsense camera calibration. Press 'c' to capture image and 'q' to quit.")

try:
    while count_images < num_images_required:
        # Wait for a coherent color frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        frame = np.asanyarray(color_frame.get_data())

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # If found, display the corners
        if ret == True:
            cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)

            # Capture image on 'c' key press
            if cv2.waitKey(1) & 0xFF == ord('c'):
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                print(f"Calibration image {count_images + 1} captured.")
                count_images += 1

        cv2.imshow('Calibration', frame)

        # Quit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop streaming
    pipeline.stop()

# Perform calibration only if enough images were captured
if len(objpoints) == num_images_required:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Calibration successful.")

    # Save the camera matrix and the distortion coefficients to a file
    np.savez(file_path, ret=ret, camera_matrix=mtx, dist_coeffs=dist, rvecs=rvecs, tvecs=tvecs)
    print("Calibration data saved.")
else:
    print("Calibration failed or insufficient images captured.")