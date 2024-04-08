import cv2
import numpy as np
import pyrealsense2 as rs

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

save_file_path = "/dev_ws/src/ur10e_examples/captures/piece_origin_aruco.jpg"

# Define the file path for camera calibration data
calibration_file_path = "/dev_ws/src/ur10e_examples/calibration/realsense_calibration_data.npz"

# Load camera calibration data
calibration_data = np.load(calibration_file_path)
mtx = calibration_data['camera_matrix']
dist = calibration_data['dist_coeffs']

# Define the real size of the ArUco marker in meters
aruco_marker_size = 0.0305  # 10 centimeters in meters

# Brown color range in HSV
brown_lower = np.array([10, 100, 20])
brown_upper = np.array([30, 255, 200])

# Wait for a coherent color frame
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()

if color_frame:
    # Convert color frame to numpy array
    frame = np.asanyarray(color_frame.get_data())

    # Convert the color frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Set dictionary size depending on the ArUco marker selected
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Detector parameters can be set here (List of detection parameters[3])
    parameters = cv2.aruco.DetectorParameters_create()
    parameters.adaptiveThreshConstant = 10

    # Lists of IDs and the corners belonging to each ID
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Convert the color frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask brown lines
    mask_brown = cv2.inRange(hsv, brown_lower, brown_upper)

    # Find contours of brown lines
    contours, _ = cv2.findContours(mask_brown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw bounding box around the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Calculate the middle points of the edges
    bottom_mid = (int((x + x + w) / 2), y + h)

    # Convert pixel coordinates to real-world coordinates
    bottom_mid_homogeneous = np.array([[bottom_mid[0]], [bottom_mid[1]], [1]])
    bottom_mid_real = np.dot(np.linalg.inv(mtx), bottom_mid_homogeneous)
    bottom_mid_real /= bottom_mid_real[2]  # Normalize homogeneous coordinates
    bottom_mid_real = bottom_mid_real[:2]  # Discard the third component

    # Draw the middle point of the bottom edge
    cv2.circle(frame, bottom_mid, 5, (255, 0, 0), -1)

    # Create a mask for the pink rectangle
    pink_rect_mask = np.zeros_like(frame)
    cv2.rectangle(pink_rect_mask, (x, y), (x + w, y + h), (255, 192, 203), -1)  # Pink color in BGR format

    # Blend the pink rectangle with the original frame
    alpha = 0.4  # Transparency (0.0 - fully transparent, 1.0 - fully opaque)
    blended_frame = cv2.addWeighted(frame, 1 - alpha, pink_rect_mask, alpha, 0)

    # Display the real x and y values of the bottom middle point
    cv2.putText(blended_frame, f"Origin Point (m): ({bottom_mid_real[0][0]:.4f}, {bottom_mid_real[1][0]:.4f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Check if the IDs list is not empty
    if np.all(ids is not None):
        for i in range(0, ids.size):
            # Estimate pose of each marker and return the values
            # rvec and tvec-different from camera coefficients
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, aruco_marker_size, mtx, dist)

            # Extract z value from tvec
            z_distance = tvec[i][0][2]

            # Draw axis for the ArUco markers
            cv2.drawFrameAxes(blended_frame, mtx, dist, rvec[i], tvec[i], aruco_marker_size)

            # Draw text showing the z value
            #cv2.putText(blended_frame, f"Z Distance: {z_distance:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Save the image with detected ArUco markers, the largest brown line, width length, and middle points
    cv2.imwrite(save_file_path, blended_frame)
    print(f"Image saved at: {save_file_path}")

# Release the RealSense pipeline
pipeline.stop()
