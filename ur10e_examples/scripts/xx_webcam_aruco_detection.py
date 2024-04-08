import cv2
import numpy as np
from cv2 import aruco

# Define the file path
file_path = "/dev_ws/src/ur10e_examples/calibration/calibration_data.npz"

calibration_data = np.load(file_path)
mtx = calibration_data['camera_matrix']
dist = calibration_data['dist_coeffs']

cap = cv2.VideoCapture(1)

while (True):
    ret, frame = cap.read()

    # operations on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # set dictionary size depending on the aruco marker selected
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    # detector parameters can be set here (List of detection parameters[3])
    parameters = aruco.DetectorParameters()
    parameters.adaptiveThreshConstant = 10

    # lists of ids and the corners belonging to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # font for displaying text (below)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # check if the ids list is not empty
    # if no check is added the code will crash
    if np.all(ids != None):

        # estimate pose of each marker and return the values
        # rvet and tvec-different from camera coefficients
        rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
        #(rvec-tvec).any() # get rid of that nasty numpy value array error

        for i in range(0, ids.size):
            # estimate pose of each marker and return the values
            # rvet and tvec-different from camera coefficients
            rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
            
            # Extract z value from tvec
            z_distance = tvec[i][0][2]

            # Draw axis for the ARUCO markers
            cv2.drawFrameAxes(frame, mtx, dist, rvec[i], tvec[i], 0.05)

            # Draw text showing the z value
            cv2.putText(frame, f"Z Distance: {z_distance:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # # draw a square around the markers
        # aruco.drawDetectedMarkers(frame, corners)

        # # code to show ids of the marker found
        # strg = ''
        # for i in range(0, ids.size):
        #     strg += str(ids[i][0])+', '

        # cv2.putText(frame, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
        
        for idss, cornerss in zip(ids, corners):

            cv2.polylines(frame, [cornerss.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA)
            cornerss = cornerss.reshape(4, 2)
            cornerss = cornerss.astype(int)
            top_right = cornerss[0].ravel()
            top_left = cornerss[1].ravel()
            bottom_right = cornerss[2].ravel()
            bottom_left = cornerss[3].ravel()
            cv2.putText(frame,f"id: {idss[0]}",top_right,cv2.FONT_HERSHEY_PLAIN,1.3,(200, 100, 0),2,cv2.LINE_AA)

    # display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()