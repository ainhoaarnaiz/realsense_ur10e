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

# Define the file path
file_path = "/dev_ws/src/ur10e_examples/calibration/realsense_calibration_data.npz"

calibration_data = np.load(file_path)
mtx = calibration_data['camera_matrix']
dist = calibration_data['dist_coeffs']

# Define the real size of the ArUco marker in meters
aruco_marker_size = 0.0305  # 10 centimeters in meters

while (True):
    # Wait for a coherent color frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    # Convert images to numpy arrays
    frame = np.asanyarray(color_frame.get_data())

    # operations on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # set dictionary size depending on the aruco marker selected
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # detector parameters can be set here (List of detection parameters[3])
    parameters = cv2.aruco.DetectorParameters_create()
    parameters.adaptiveThreshConstant = 10

    # lists of ids and the corners belonging to each id
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # font for displaying text (below)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # check if the ids list is not empty
    # if no check is added the code will crash
    if np.all(ids != None):
        for i in range(0, ids.size):
            # estimate pose of each marker and return the values
            # rvec and tvec-different from camera coefficients
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, aruco_marker_size, mtx, dist)
            
            # Extract z value from tvec
            z_distance = tvec[i][0][2]

            # Draw axis for the ARUCO markers
            cv2.drawFrameAxes(frame, mtx, dist, rvec[i], tvec[i], aruco_marker_size)

            # Draw text showing the z value
            cv2.putText(frame, f"Z Distance: {z_distance:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        for idss, cornerss in zip(ids, corners):
            cv2.polylines(frame, [cornerss.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA)
            cornerss = cornerss.reshape(4, 2)
            cornerss = cornerss.astype(int)
            top_right = cornerss[0].ravel()
            cv2.putText(frame, f"id: {idss[0]}", tuple(top_right), cv2.FONT_HERSHEY_PLAIN, 1.3, (200, 100, 0), 2, cv2.LINE_AA)

    # display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the RealSense pipeline
pipeline.stop()
cv2.destroyAllWindows()
