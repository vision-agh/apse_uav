import cv2
from cv2 import aruco
import numpy as np
import json
import csv
import os
from scipy.spatial.transform import Rotation as R

#%%====================================
#PARAMETERS TO BE CHANGED BY USER

#index of first frame to be processed
start_frame = 1
#index of last frame to be processed, if None: all frames from input folder/input video folder will be processed
#you can also terminate the processing immediately by press 'q' key
stop_frame = None
#change the value if you want to skip some frames on the sequence
step_frame = 1

#True if you want to show image with results, False otherwise
showImage = True
#value for cv2.waitKey() function - 0: wait for key to be pressed, otherwise: time in miliseconds to show image
cv2waitKeyVal = 1

#True if you want to save the results to a file, False otherwise
saveResults = False
#True if you want to save images after detection on the disk, False otherwise
saveImages = False

#True if you use data from DCNN, False if you only use Aruco method
useCentroidData = False

#number of frames to be used for marker size averaging, recommended is 1
N_avg = 1

#True if you want to draw markers on image, False otherwise
drawMarkers = True
#True if you want to draw axes of the markers on image, False otherwise
drawMarkersAxes = False
#True if you want to print pose and ID of the markers on image, False otherwise
showDataOnImage = False
#True if you want to print distances between vehicles on image, False otherwise
showDistancesOnImage = True

#True if you want to draw LEDs of the host car, False otherwise
drawLeds = False
#threshold value for LEDs detection - None: use default value (190 + altitude in metres), 0-255: your value
LEDs_threshold = None

#True if you want Lidar to be the source of measurements, False if you want host's Aruco marker
sourceLidar = False
#True if you want to draw lines from Lidar/host's Aruco to vehicles, False otherwise
#colour info: distance to Aruco marker - red, distance to closest point - yellow
drawLines = True
#True if you want to draw points on the image, False otherwise
#colour info: Aruco centroid and Lidar - cyan, DCNN centroid - magenta, DCNN closest point - white
drawPoints = False

#path to camera parameters file
path_camera_params = "your_path" + "cam_params.json"

#True if you use images as input, False if you use video
useImages = False
#path to folder with input images
#images inside must be named image_XXXX.png, where XXXX is the frame number
if useImages:
    path_input_images = "your_path"

#True if you use video as input, False if you you images
useVideo = True
#path to an input video (path + filename + extension)
if useVideo:
    path_input_video = "your_path"

#path to data from DCNN detection, used only if useCentroidData is True (path + filename.csv)
if useCentroidData:
    path_dcnn_data = "your_path"

#path to save results to a file, used only if saveResults is True (path + filename.csv)
#be careful not to overwrite any existing file!
if saveResults:
    path_output_results = "your_path"

#path to save images to a folder, used only if saveImages is True
#path must lead to an existing folder!
if saveImages:
    path_output_images = "your_path"

#%%====================================
#FUNCTIONS FOR DATA INPUT/OUTPUT

def readCameraParams():
    #read camera parameters from file
    with open(path_camera_params, "r") as file:
        cam_params = json.load(file)
        
    #camera matrix
    mtx = np.array(cam_params["mtx"])
    
    #distortion coefficients
    dist = np.array(cam_params["dist"])
    
    return mtx, dist

def readCentroidData(path_dcnn_data):
    #open data file with centroids and bboxes from DCNN detection and store it in centroid_data variable
    centroid_data = []    
    
    with open(path_dcnn_data) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 1:
                temp = []
                for i in range(17):
                    if row[i] == '' or row[i] == 'nan':
                        row[i] = 0
                    temp.append(int(row[i]))
                centroid_data.append(temp)
            line_count += 1    
    csv_file.close()
    
    return centroid_data

def outputDataInit():
    #clear output file
    file = open(path_output_results, "w")
    
    #write names of the columns for data
    if useCentroidData:
        file.write("frame_ID ,ID_4_detected ,markerLength ,leds_ID ,UAV_altitude ,fov_width ,fov_height ," +
                "ID_1_detected ,distance_veh1_aruco ,distance_veh1_aruco_bbox ,distance_veh1_dcnn ,distance_veh1_dcnn_bbox ," +
                "ID_2_detected ,distance_veh2_aruco ,distance_veh2_aruco_bbox ,distance_veh2_dcnn ,distance_veh2_dcnn_bbox ," +
                "ID_3_detected ,distance_veh3_aruco ,distance_veh3_aruco_bbox ,distance_veh3_dcnn ,distance_veh3_dcnn_bbox" + "\n")
    else:
        file.write("frame_ID ,ID_4_detected ,markerLength ,leds_ID ,UAV_altitude ,fov_width ,fov_height ," +
                "ID_1_detected ,distance_veh1_aruco ,distance_veh1_aruco_bbox ," +
                "ID_2_detected ,distance_veh2_aruco ,distance_veh2_aruco_bbox ," +
                "ID_3_detected ,distance_veh3_aruco ,distance_veh3_aruco_bbox ," + "\n")
    
    file.close()
    file = open(path_output_results, "a")
    
    return file
    
def outputData(file):
    #temp values to write if particular markers were not detected (then distances = 0)
    fill_with_zeros3 = "," + str(0) + "," + str(0) + "," + str(0)
    fill_with_zeros5 = "," + str(0) + "," + str(0) + "," + str(0) + "," + str(0) + "," + str(0)
    
    #write frame number and if vehicle 4 was detected
    file.write(str(k) + "," + str(detected_ID[3]))
    
    #for detected vehicle 4 write marker length in metres, LEDs ID, altitude of the UAV in metres (estimated from vision method), estimated FOV of the camera in metres
    if(detected_ID[3] == 1):
        file.write("," + str(round(markerLength,5)) + "," + str(leds) + "," + str(round(altitude,2)) + "," + str(round(width*markerLength/msp4,2)) + "," + str(round(height*markerLength/msp4,2)))
    else:
        file.write(fill_with_zeros5)
    
    #for other vehicles write if they were detected, distance to Aruco marker and bounding box and - if used - same points from DCNN detection
    if(detected_ID[0] == 1): #vehicle 1
        if useCentroidData:
            file.write("," + str(detected_ID[0]) + "," + str(round(dist_veh1_aruco,3)) + "," + str(round(dist_veh1_aruco_bbox,3)) + "," + str(round(dist_veh1_dcnn,3)) + "," + str(round(dist_veh1_dcnn_bbox,3)))
        else:
            file.write("," + str(detected_ID[0]) + "," + str(round(dist_veh1_aruco,3)) + "," + str(round(dist_veh1_aruco_bbox,3)))
    else:
        file.write(fill_with_zeros5 if useCentroidData else fill_with_zeros3)
    
    if(detected_ID[1] == 1): #vehicle 2
        if useCentroidData:
            file.write("," + str(detected_ID[1]) + "," + str(round(dist_veh2_aruco,3)) + "," + str(round(dist_veh2_aruco_bbox,3)) + "," + str(round(dist_veh2_dcnn,3)) + "," + str(round(dist_veh2_dcnn_bbox,3)))
        else:
            file.write("," + str(detected_ID[1]) + "," + str(round(dist_veh2_aruco,3)) + "," + str(round(dist_veh2_aruco_bbox,3)))
    else:
        file.write(fill_with_zeros5 if useCentroidData else fill_with_zeros3)
    
    if(detected_ID[2] == 1): #vehicle 3
        if useCentroidData:
            file.write("," + str(detected_ID[2]) + "," + str(round(dist_veh3_aruco,3)) + "," + str(round(dist_veh3_aruco_bbox,3)) + "," + str(round(dist_veh3_dcnn,3)) + "," + str(round(dist_veh3_dcnn_bbox,3)))
        else:
            file.write("," + str(detected_ID[2]) + "," + str(round(dist_veh3_aruco,3)) + "," + str(round(dist_veh3_aruco_bbox,3)))
    else:
        file.write(fill_with_zeros5 if useCentroidData else fill_with_zeros3)

    file.write("\n")

#%%====================================
#FUNCTIONS FOR SETTING PARAMETERS

def setArucoParameters():
    parameters = aruco.DetectorParameters_create()
    
    #set values for Aruco detection parameters
    parameters.minMarkerPerimeterRate = 0.01 #enables detection from higher altitude
    parameters.perspectiveRemovePixelPerCell = 8
    parameters.perspectiveRemoveIgnoredMarginPerCell = 0.33
    parameters.errorCorrectionRate = 2.0 #much more detections from high altitude, but FP happen sometimes
    parameters.aprilTagMinClusterPixels = 100 #less candidates to encode ID
    parameters.aprilTagMaxNmaxima = 5
    parameters.aprilTagCriticalRad = 20*np.pi/180 #much less candidates to encode ID
    parameters.aprilTagMaxLineFitMse = 1
    parameters.aprilTagMinWhiteBlackDiff = 100 #faster detection, but in bad contrast problems may happen
    #parameters.aprilTagQuadDecimate = 1.5 #huge detection time speedup, but at the cost of fewer detections and worse accuracy
    
    #default set of all Aruco detection parameters
    #parameters.adaptiveThreshWinSizeMin = 3
    #parameters.adaptiveThreshWinSizeMax = 23
    #parameters.adaptiveThreshWinSizeStep = 10
    #parameters.adaptiveThreshConstant = 7
    #parameters.minMarkerPerimeterRate = 0.03
    #parameters.maxMarkerPerimeterRate = 4
    #parameters.polygonalApproxAccuracyRate = 0.03
    #parameters.minCornerDistanceRate = 0.05
    #parameters.minDistanceToBorder = 3
    #parameters.minMarkerDistanceRate = 0.05
    #parameters.cornerRefinementMethod = aruco.CORNER_REFINE_NONE
    #parameters.cornerRefinementWinSize = 5
    #parameters.cornerRefinementMaxIterations = 30
    #parameters.cornerRefinementMinAccuracy = 0.1
    #parameters.markerBorderBits = 1
    #parameters.perspectiveRemovePixelPerCell = 4
    #parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
    #parameters.maxErroneousBitsInBorderRate = 0.35
    #parameters.minOtsuStdDev = 5.0
    #parameters.errorCorrectionRate = 0.6    
    #parameters.aprilTagMinClusterPixels = 5
    #parameters.aprilTagMaxNmaxima = 10
    #parameters.aprilTagCriticalRad = 10*np.pi/180
    #parameters.aprilTagMaxLineFitMse = 10
    #parameters.aprilTagMinWhiteBlackDiff = 5
    #parameters.aprilTagDeglitch = 0
    #parameters.aprilTagQuadDecimate = 0
    #parameters.aprilTagQuadSigma = 0
    #parameters.detectInvertedMarker = False
    
    return parameters

def setAverageMarkerSize():
    #temp variables for averaging marker size
    msp1_avg = np.zeros((N_avg,1))
    msp2_avg = np.zeros((N_avg,1))
    msp3_avg = np.zeros((N_avg,1))
    msp4_avg = np.zeros((N_avg,1))
    
    return msp1_avg, msp2_avg, msp3_avg, msp4_avg

#%%====================================
#FUNCTIONS FOR ARUCO MARKERS

def preprocessFrame(frame):
    #remove distortion from camera
    frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    
    #perform gamma correction
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    lab[...,0] = cv2.LUT(lab[...,0], lookUpTable)
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return frame

def detectArucoMarkers(gray, parameters):
    #use predefined Aruco markers dictionary
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    
    #detect markers with APRILTAG method
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters, cameraMatrix=mtx, distCoeff=dist)

    return corners, ids

def getMarkerData(corners, rvec, cx_prev, cy_prev):
    #marker centre x and y
    cx = int(corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4
    cy = int(corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4
    
    #marker size in pixels, cosine of yaw angle, sine of yaw angle
    msp = ((np.sqrt(np.power((corners[1][0]-corners[0][0]),2) + np.power((corners[1][1]-corners[0][1]),2)) + 
               np.sqrt(np.power((corners[2][0]-corners[1][0]),2) + np.power((corners[2][1]-corners[1][1]),2)) + 
               np.sqrt(np.power((corners[3][0]-corners[2][0]),2) + np.power((corners[3][1]-corners[2][1]),2)) + 
               np.sqrt(np.power((corners[0][0]-corners[3][0]),2) + np.power((corners[0][1]-corners[3][1]),2))) / 4)
        
    #distance in metres between marker of the same ID on subsequent frames
    if cx_prev is not None and cy_prev is not None:
        diff = np.sqrt(np.power(cx_prev-cx,2) + np.power(cy_prev-cy,2)) * markerLength / msp
    else:
        diff = 0
    
    return abs(cx), abs(cy), msp, diff

def calculateAverageMarkerSize(msp_avg, msp):
    #write last measured marker size into table
    if(N_avg == 1):
        msp_avg = msp
    elif(N_avg > 1 and isinstance(N_avg, int)):
        for j in range(N_avg-1):
            msp_avg[j] = msp_avg[j+1]
        msp_avg[N_avg-1] = msp
    
    #calculate the average and rescale marker size
    nonzero = np.count_nonzero(msp_avg)
    size_corr = np.sum(msp_avg)/(msp*nonzero)
    msp = msp * size_corr
    
    return size_corr, msp

def markerLengthCorrection(altitude):
    #use correction of marker size based on current altitude
    return markerLengthOrg * (1 - 0.00057 * altitude/marker_div) / div

def printDataOnImage(corners, tvec, rvec, ids):
    font = cv2.FONT_HERSHEY_SIMPLEX
    r = R.from_rotvec(rvec)    
    
    #calculate real altitude to be printed
    tvec_temp = tvec.copy()
    tvec_temp[2] = tvec_temp[2]/marker_div
    
    #calculate angles and position and convert them to text
    ang = 'R = ' + str([round(r.as_euler('zxy', degrees=True)[0],2),
                        round(r.as_euler('zxy', degrees=True)[1],2),
                        round(r.as_euler('zxy', degrees=True)[2],2)]) + 'deg' 
    pos = 't = ' + str([round(j,3) for j in tvec_temp]) + 'm'  
    id = 'ID = ' + str(ids)    
    
    #calculate the position where the text will be placed on image
    position = tuple([int(corners[0]-150), int(corners[1]+150)])
    position_ang = tuple([int(position[0]-0), int(position[1]+50)])
    position_id = tuple([int(position[0]-0), int(position[1]-50)])
    
    #write the text onto the image
    cv2.putText(frame, id, position_id, font, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, pos, position, font, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, ang, position_ang, font, 1.4, (255, 255, 255), 2, cv2.LINE_AA)

#%%====================================
#FUNCTIONS FOR POINTS CALCULATIONS

def detectAndDrawLEDs(gray, tvec, rvec, size_corr, msp, threshold = None):
    #position of the LEDs wrt. Aruco marker
    axis_leds = np.float32([[-0.419,-0.42,0],[-0.414,-0.305,0],[-0.409,-0.19,0],[-0.404,-0.07,0],
                            [-0.399,0.065,0],[-0.393,0.19,0],[-0.388,0.315,0],[-0.382,0.435,0]])
    
    #project these points onto the image
    imgpts_leds, _ = cv2.projectPoints(axis_leds, rvec, tvec/size_corr, mtx, dist)
    imgpts_leds = np.maximum(0,np.int32(imgpts_leds).reshape(-1,2))
    
    #use 190 + altitude in metres as the default value if the user did not specify the threshold
    thr = max(190 + int(tvec[2]/marker_div), 240) if threshold is None else threshold
    
    value = ''
    leds = 0
    for j in range(8):
        x = int(imgpts_leds[j][0])
        y = int(imgpts_leds[j][1])
        
        #use 5x5 neighbourhood of pixels
        point = gray[y-2:y+3,x-2:x+3]
        val = np.sum(np.sum(point))/25
        
        #if the LED is on
        if val > thr:
            value = value + '1'
            leds = leds + np.power(2,7-j)
            if drawLeds:
                cv2.circle(frame, (x,y), int(msp/15)+1, color=(0,255,0), thickness=int(msp/30)+1)
        
        #if the LED is off
        else:
            value = value + '0'
            if drawLeds:
                cv2.circle(frame, (x,y), int(msp/15)+1, color=(0,0,255), thickness=int(msp/30)+1)

    return leds

def centroidFromAruco(veh_coords, tvec, rvec, size_corr):
    #project measured centroid of the vehicle wrt. Aruco marker onto image
    imgpts, _ = cv2.projectPoints(veh_coords, rvec, tvec/size_corr, mtx, dist)
    imgpts = np.maximum(0,np.int32(imgpts).reshape(-1,2))
    if drawPoints:
        cv2.circle(frame, (int(imgpts[0][0]),int(imgpts[0][1])), 5, color=(255,255,0), thickness=-1)
        
    return imgpts

def centroidFromDCNN(centroid_data_x, centroid_data_y):
    #use the centroid of the vehicle from DCNN detection
    xc = centroid_data_x
    yc = centroid_data_y
    
    #set and draw the point on the image
    imgpts = np.maximum(0,np.int32(np.array([[xc, yc, 0]])))
    if drawPoints:
        cv2.circle(frame, (int(imgpts[0][0]),int(imgpts[0][1])), 5, color=(255,0,255), thickness=-1)
    
    return imgpts

def boundingBoxFromDCNN(centroid_data_x, centroid_data_y):
    #use the closest point of the vehicle from DCNN detection
    xc = centroid_data_x
    yc = centroid_data_y
    imgpts = np.maximum(0,np.int32(np.array([[xc, yc, 0]])))
    if drawPoints:
        cv2.circle(frame, (int(imgpts[0][0]),int(imgpts[0][1])), 5, color=(255,255,255), thickness=-1)
    
    return imgpts

def drawBoundingBox(tvec, rvec, veh_dim, size_corr):
    #calculate angles in horizontal and vertical direction
    alpha_h = np.arctan(tvec[0]/tvec[2])
    alpha_v = np.arctan(tvec[1]/tvec[2])
    
    #calucalate yaw angle of the vehicle
    r = R.from_rotvec(rvec[0])
    yaw = round(r.as_euler('zxy', degrees=True)[0],2)
    
    #based on yaw angle of the vehicle, alpha angles may be negative
    alpha_h = alpha_h if yaw < 0 else -alpha_h
    alpha_v = alpha_v if yaw < 0 else -alpha_v
    
    #modify dimensions of vehicle's bbox
    veh_dim = np.multiply(veh_dim, [1-alpha_h/2, 1+alpha_h/2, 1-alpha_v/2, 1+alpha_v/2])
    
    #use modified values to set corners of bbox, project these points onto the image and draw bbox
    axis = np.float32([[veh_dim[2],veh_dim[0],0], [veh_dim[2],veh_dim[1],0], [veh_dim[3],veh_dim[1],0], [veh_dim[3],veh_dim[0],0]])
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec/size_corr, mtx, dist)
    imgpts = np.maximum(0,np.int32(imgpts).reshape(-1,2))
    cv2.drawContours(frame, [imgpts[0:4]], -1, (255,0,0), 5)

    return veh_dim

#%%====================================
#FUNCTIONS FOR DISTANCE CALCULATION

def generatePointsBoundingBox(veh_dim):
    #generate additional points on bounding box - 20 along the length and 8 along the width of the vehicle
    points_l = 20
    points_w = 8
    
    o1 = np.linspace(veh_dim[0], veh_dim[1], points_l)
    o2 = np.linspace(veh_dim[2], veh_dim[3], points_w)
    
    object1 = np.zeros((points_l,2))
    object2 = np.zeros((points_l,2))
    object3 = np.zeros((points_w,2))
    object4 = np.zeros((points_w,2))
    
    object1[:,0] = o1
    object1[:,1] = veh_dim[2]
    object2[:,0] = o1
    object2[:,1] = veh_dim[3]
    object3[:,0] = veh_dim[0]
    object3[:,1] = o2
    object4[:,0] = veh_dim[1]
    object4[:,1] = o2
    
    #concatenate the points generated on each edge of bbox    
    object = np.concatenate((object1, object2, object3, object4))
    w, h = object.shape
    bbox = np.zeros((w, h+1))
    
    bbox[:,0] = object[:,1]
    bbox[:,1] = object[:,0]
    bbox[:,2] = 0
    
    return bbox

def findMinimumDistanceBoundingBox(source, bbox, tvec, rvec, size_corr):
    #project generated bbox points onto image
    imgpts, _ = cv2.projectPoints(bbox, rvec, tvec/size_corr, mtx, dist)
    imgpts = np.maximum(0,np.int32(imgpts).reshape(-1,2))

    #find minimum distance between source of signal and generated bbox points
    distance = np.inf
    index = 0
    for i in range(len(imgpts)):
        d = np.sqrt(pow(source[0][0]-imgpts[i][0],2) + pow(source[0][1]-imgpts[i][1],2))
        if(d < distance):
            distance = d
            index = i
    
    #return the closest point
    return imgpts[index]

def calculateDistance(lidar, aruco, bbox, markerLength, msp4, msp):
    #calculate distances to Aruco marker and bbox of the vehicle
    d_aruco = np.sqrt((lidar[0][0]-aruco[0][0]) * (lidar[0][0]-aruco[0][0]) + (lidar[0][1]-aruco[0][1]) * (lidar[0][1]-aruco[0][1]))
    d_bbox = np.sqrt((lidar[0][0]-bbox[0][0]) * (lidar[0][0]-bbox[0][0]) + (lidar[0][1]-bbox[0][1]) * (lidar[0][1]-bbox[0][1]))
    
    #convert distances from pixels to metres
    dist_aruco = d_aruco * markerLength / ((msp4+msp)/2)
    dist_bbox = d_bbox * markerLength / ((msp4+msp)/2)
    
    return dist_aruco, dist_bbox

def drawLinesOnImage(source, point, cx, cy, dist_aruco, dist_aruco_bbox, veh_id):
    #draw the line from source of the measurement to the closest point of the vehicle
    cv2.line(frame, (int(source[0][0]), int(source[0][1])), (int(point[0]), int(point[1])), (0,255,255), 5)
    
    #draw the line from source of the measurement to the centre of vehicle' Aruco marker
    cv2.line(frame, (int(source[0][0]), int(source[0][1])), (int(cx), int(cy)), (0,0,255), 5)
    
    if showDistancesOnImage:
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        #calculate angles and position and convert them to text
        dist_aruco = str(round(dist_aruco,1)) + ','
        dist_aruco_bbox = str(round(dist_aruco_bbox,1)) + ' m'
        
        #calculate the position where the text will be placed on image
        position_red = tuple([int((source[0][0]+cx)/2-200), int((source[0][1]+cy)/2)-50])
        position_yellow = tuple([int((source[0][0]+cx)/2+50), int((source[0][1]+cy)/2)-50])
        
        #write the text onto the image
        cv2.putText(frame, dist_aruco, position_red, font, 3.0, (0, 0, 255), 6, cv2.LINE_AA)
        cv2.putText(frame, dist_aruco_bbox, position_yellow, font, 3.0, (0, 255, 255), 6, cv2.LINE_AA)

#%%====================================
#ALGORITHM PARAMETERS (DO NOT CHANGE!) AND DATA READ

height, width = 2160, 3840 #fixed input image/video resolution
markerLengthOrg = 0.55 #real size of the marker in metres, this value does not change in algorithm
markerLength = markerLengthOrg #real size of the marker in metres, this value changes in algorithm
marker_div = 1.2 #correction for altitude estimation from marker
div = 1.013 #additional correction for distance calculation (based on altitude test)
DIFF_MAX = 2/3 * step_frame * 2 #maximum displacement of ArUco centre between frames with vehicle speed of 72 km/h = 20 m/s

if useCentroidData:
    centroid_data = readCentroidData(path_dcnn_data) #read centroid data from DCNN
if saveResults:
    file = outputDataInit() #initialize output file for saving results
    
parameters = setArucoParameters() #create Aruco detection parameters
mtx, dist = readCameraParams() #read camera parameters
msp1_avg, msp2_avg, msp3_avg, msp4_avg = setAverageMarkerSize() #initialization of marker size averaging variables
detected_ID_prev = [0,0,0,0] #initialization of vehicle detection state on previous frame
[cx1_prev, cy1_prev, cx2_prev, cy2_prev, cx3_prev, cy3_prev, cx4_prev, cy4_prev] = np.zeros(8, dtype='int') #initialization of ArUco marker centres

gamma = 2 #gamma parameter value
lookUpTable = np.empty((1,256), np.uint8) #look-up table for gamma correction
for i in range(256):
    lookUpTable[0,i] = np.clip(pow(i/255.0, gamma) * 255.0, 0, 255)

#host vehicle's Lidar wrt. Aruco marker in metres
veh4_coords_lidar = np.float32([[-0.05,-0.80,0]])

#vehicle's centroid wrt. Aruco marker in metres
veh4_coords = np.float32([[0,0.07,0]])
veh1_coords = np.float32([[0,0.42,0]])
veh2_coords = np.float32([[0,0.59,0]])
veh3_coords = np.float32([[0,0.58,0]])

#initialize values if images are used
if useImages:
    k = start_frame
    stop_frame = len(os.listdir(path_input_images)) if stop_frame is None else stop_frame

#initialize values if video is used
elif useVideo:
    video = cv2.VideoCapture(path_input_video)
    k = start_frame
    if start_frame > 1 and video.isOpened():
        for i in range(start_frame-1):
            ret, frame = video.read()
            if ret == False:
                break
    stop_frame = np.inf if stop_frame is None else stop_frame

#calculate maps for undistortion
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (width, height), 5)

#iterate over frames
while k <= stop_frame and (useImages or (useVideo and video.isOpened())):
    #read frame from image or video
    if useImages:
        frame = cv2.imread(path_input_images + "image_%04d.png" % k)
    elif useVideo:
        ret, frame = video.read()
        if ret == False:
            break

    detected_ID = [0,0,0,0] #by default no vehicle is detected in image

    #real vehicle dimensions in metres wrt. Aruco marker: back, front, left, right
    veh4_dim = [-2.35, 2.49, -0.86, 0.86]
    veh1_dim = [-1.95, 2.8, -0.9, 0.9]
    veh2_dim = [-1.68, 2.86, -0.87, 0.87]
    veh3_dim = [-1.32, 2.48, -0.86, 0.86]    

    #frame preprocessing - camera distortion removal and gamma correction
    frame = preprocessFrame(frame)

    #convert image to grayscale and detect Aruco markers
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids = detectArucoMarkers(gray, parameters)

#%%====================================
#MARKER DETECTION AND POINTS CALCULATIONS

    #if any marker was detected
    if np.all(ids != None):
        #estimate pose of detected markers
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, markerLength, mtx, dist)
        
        #iterate over all detected markers
        for i in range(len(ids)):
            #only markers with ID={1,2,3,4} are used at this moment
            if(ids[i][0] == 4): #vehicle 4 (host)
                cx4, cy4, msp, diff4 = getMarkerData(corners[i][0], rvec[i][0], None if k == start_frame else cx4_prev, None if k == start_frame else cy4_prev) #get detected marker parameters
                
                if detected_ID_prev[3] == 0: #if this marker was not detected on previous frame, it may be 'new' or FP
                    detected_ID[3] = 1 #mark vehicle as detected 
                    cx4_prev, cy4_prev = cx4, cy4 #save position of the marker in the image
                
                if (detected_ID_prev[3] == 1 and diff4 < DIFF_MAX) or k == start_frame: #if this marker was detected on previous frame and its position in the image is similar
                    if drawMarkers:
                        cv2.drawContours(frame, [np.maximum(0,np.int32(corners[i][0]))], -1, (0,255,0), 3)
                    if drawMarkersAxes:
                        aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], markerLength)
                    if showDataOnImage:
                        printDataOnImage(corners[i][0][0], tvec[i][0], rvec[i][0], ids[i][0])
                    
                    detected_ID[3] = 1 #mark vehicle as detected
                    altitude = tvec[i][0][2] #altitude info
                    markerLength = markerLengthCorrection(altitude) #correction of original marker size based on altitude
                    altitude = altitude/marker_div #calculate real altitude
                    
                    size_corr4, msp4 = calculateAverageMarkerSize(msp4_avg, msp) #marker size averaging
                    leds = detectAndDrawLEDs(gray, tvec[i][0], rvec[i], size_corr4, msp4, LEDs_threshold) #LEDs detection
                    
                    imgpts_veh4 = centroidFromAruco(veh4_coords, tvec[i], rvec[i], size_corr4) #calculate centroid of the vehicle wrt. Aruco marker
                    imgpts_veh4_lidar = centroidFromAruco(veh4_coords_lidar, tvec[i], rvec[i], size_corr4) #calculate Lidar's position wrt. Aruco marker
                    cx4_prev, cy4_prev = cx4, cy4 #save position of the marker in the image
                    
                    if useCentroidData:
                        imgpts_veh4_dcnn = centroidFromDCNN(centroid_data[k-1][1], centroid_data[k-1][2]) #calculate Aruco position wrt. vehicle centroid from DCNN
                    veh4_dim = drawBoundingBox(tvec[i][0], rvec[i], veh4_dim, size_corr4) #draw bounding box of the vehicle
                else: #detected marker is a FP, change its ID to incorrect value
                    ids[i][0] = -1
            
            if([4] not in ids): #if host is not detected, use altitude data from another marker
                altitude = tvec[i][0][2] #altitude info
                markerLength = markerLengthCorrection(altitude) #correction of original marker size based on altitude
                altitude = altitude/marker_div #calculate real altitude
            
            if(ids[i][0] == 1): #vehicle 1
                cx1, cy1, msp, diff1 = getMarkerData(corners[i][0], rvec[i][0], None if k == start_frame else cx1_prev, None if k == start_frame else cy1_prev) #get detected marker parameters
                
                if detected_ID_prev[0] == 0: #if this marker was not detected on previous frame, it may be 'new' or FP
                    detected_ID[0] = 1 #mark vehicle as detected
                    cx1_prev, cy1_prev = cx1, cy1 #save position of the marker in the image
                    
                if (detected_ID_prev[0] == 1 and diff1 < DIFF_MAX) or k == start_frame: #if this marker was detected on previous frame and its position in the image is similar
                    if drawMarkers:
                        cv2.drawContours(frame, [np.maximum(0,np.int32(corners[i][0]))], -1, (0,255,0), 3)
                    if drawMarkersAxes:
                        aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], markerLength)
                    if showDataOnImage:
                        printDataOnImage(corners[i][0][0], tvec[i][0], rvec[i][0], ids[i][0])
                    
                    detected_ID[0] = 1 #mark vehicle as detected
                    size_corr1, msp1 = calculateAverageMarkerSize(msp1_avg, msp) #marker size averaging
                    imgpts_veh1 = centroidFromAruco(veh1_coords, tvec[i], rvec[i], size_corr1) #calculate centroid of the vehicle wrt. Aruco marker
                    cx1_prev, cy1_prev = cx1, cy1 #save position of the marker in the image
                    
                    if useCentroidData:
                        imgpts_veh1_dcnn = centroidFromDCNN(centroid_data[k-1][5], centroid_data[k-1][6]) #calculate Aruco position wrt. vehicle centroid from DCNN
                        imgpts_veh1_dcnn_bbox = boundingBoxFromDCNN(centroid_data[k-1][7], centroid_data[k-1][8]) #calculate closest point of the vehicle from DCNN
                    veh1_dim = drawBoundingBox(tvec[i][0], rvec[i], veh1_dim, size_corr1) #draw bounding box of the vehicle
                else: #detected marker is a FP, change its ID to incorrect value
                    ids[i][0] = -1
                
            if(ids[i][0] == 2): #vehicle 2
                cx2, cy2, msp, diff2 = getMarkerData(corners[i][0], rvec[i][0], None if k == start_frame else cx2_prev, None if k == start_frame else cy2_prev) #get detected marker parameters
                
                if detected_ID_prev[1] == 0: #if this marker was not detected on previous frame, it may be 'new' or FP
                    detected_ID[1] = 1 #mark vehicle as detected
                    cx2_prev, cy2_prev = cx2, cy2 #save position of the marker in the image
                
                if (detected_ID_prev[1] == 1 and diff2 < DIFF_MAX) or k == start_frame: #if this marker was detected on previous frame and its position in the image is similar
                    if drawMarkers:
                        cv2.drawContours(frame, [np.maximum(0,np.int32(corners[i][0]))], -1, (0,255,0), 3)
                    if drawMarkersAxes:
                        aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], markerLength)
                    if showDataOnImage:
                        printDataOnImage(corners[i][0][0], tvec[i][0], rvec[i][0], ids[i][0])
                    
                    detected_ID[1] = 1 #mark vehicle as detected
                    size_corr2, msp2 = calculateAverageMarkerSize(msp2_avg, msp) #marker size averaging
                    imgpts_veh2 = centroidFromAruco(veh2_coords, tvec[i], rvec[i], size_corr2) #calculate centroid of the vehicle wrt. Aruco marker
                    cx2_prev, cy2_prev = cx2, cy2 #save position of the marker in the image
                    
                    if useCentroidData:
                        imgpts_veh2_dcnn = centroidFromDCNN(centroid_data[k-1][9], centroid_data[k-1][10]) #calculate Aruco position wrt. vehicle centroid from DCNN
                        imgpts_veh2_dcnn_bbox = boundingBoxFromDCNN(centroid_data[k-1][11], centroid_data[k-1][12]) #calculate closest point of the vehicle from DCNN
                    veh2_dim = drawBoundingBox(tvec[i][0], rvec[i], veh2_dim, size_corr2) #draw bounding box of the vehicle
                else: #detected marker is a FP, change its ID to incorrect value
                    ids[i][0] = -1
            
            if(ids[i][0] == 3): #vehicle 3
                cx3, cy3, msp, diff3 = getMarkerData(corners[i][0], rvec[i][0], None if k == start_frame else cx3_prev, None if k == start_frame else cy3_prev) #get detected marker parameters
                
                if detected_ID_prev[2] == 0: #if this marker was not detected on previous frame, it may be 'new' or FP
                    detected_ID[2] = 1 #mark vehicle as detected 
                    cx3_prev, cy3_prev = cx3, cy3 #save position of the marker in the image
                
                if (detected_ID_prev[2] == 1 and diff3 < DIFF_MAX) or k == start_frame: #if this marker was detected on previous frame and its position in the image is similar
                    if drawMarkers:
                        cv2.drawContours(frame, [np.maximum(0,np.int32(corners[i][0]))], -1, (0,255,0), 3)
                    if drawMarkersAxes:
                        aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], markerLength)
                    if showDataOnImage:
                        printDataOnImage(corners[i][0][0], tvec[i][0], rvec[i][0], ids[i][0])
                    
                    detected_ID[2] = 1 #mark vehicle as detected
                    size_corr3, msp3 = calculateAverageMarkerSize(msp3_avg, msp) #marker size averaging
                    imgpts_veh3 = centroidFromAruco(veh3_coords, tvec[i], rvec[i], size_corr3) #calculate centroid of the vehicle wrt. Aruco marker
                    cx3_prev, cy3_prev = cx3, cy3 #save position of the marker in the image
                    
                    if useCentroidData:
                        imgpts_veh3_dcnn = centroidFromDCNN(centroid_data[k-1][13], centroid_data[k-1][14]) #calculate Aruco position wrt. vehicle centroid from DCNN
                        imgpts_veh3_dcnn_bbox = boundingBoxFromDCNN(centroid_data[k-1][15], centroid_data[k-1][16]) #calculate closest point of the vehicle from DCNN
                    veh3_dim = drawBoundingBox(tvec[i][0], rvec[i], veh3_dim, size_corr3) #draw bounding box of the vehicle
                else: #detected marker is a FP, change its ID to incorrect value
                    ids[i][0] = -1

#%%====================================
#DISTANCE CALCULATION FOR VEHICLES

        #iterate again over all detected markers to use results from current frame
        for i in range(len(ids)):
            if(ids[i][0] == 4): #get host car marker ID
                #iterate over all markers to calculate distances to them from host
                for j in range(len(ids)):
                    if(ids[j][0] == 1): #vehicle 1
                        #start = time.time_ns()
                        if (detected_ID_prev[0] == 1 and diff1 < DIFF_MAX) or k == start_frame: #if this marker was detected on previous frame and its position in the image is similar
                            bbox = generatePointsBoundingBox(veh1_dim) #generate additional points for bounding box
                            if sourceLidar:
                                point = findMinimumDistanceBoundingBox(imgpts_veh4_lidar, bbox, tvec[j], rvec[j], size_corr1) #find the closest point of the bbox from Lidar
                                dist_veh1_aruco, dist_veh1_aruco_bbox = calculateDistance(imgpts_veh4_lidar, np.float32([[cx1, cy1]]), [point], markerLength, msp4, msp1) #calculate distances in metres for Aruco method
                                if drawLines:
                                    drawLinesOnImage(imgpts_veh4_lidar, point, cx1, cy1, dist_veh1_aruco, dist_veh1_aruco_bbox, ids[j][0]) #draw lines between Lidar and vehicle
                            else:
                                point = findMinimumDistanceBoundingBox(np.float32([[cx4, cy4]]), bbox, tvec[j], rvec[j], size_corr1) #find the closest point of the bbox from host's Aruco
                                dist_veh1_aruco, dist_veh1_aruco_bbox = calculateDistance(np.float32([[cx4, cy4]]), np.float32([[cx1, cy1]]), [point], markerLength, msp4, msp1) #calculate distances in metres for Aruco method
                                if drawLines:
                                    drawLinesOnImage(np.float32([[cx4, cy4]]), point, cx1, cy1, dist_veh1_aruco, dist_veh1_aruco_bbox, ids[j][0]) #draw lines between host's Aruco and vehicle
                            if useCentroidData:
                                dist_veh1_dcnn, dist_veh1_dcnn_bbox = calculateDistance(imgpts_veh4_lidar, imgpts_veh1_dcnn, imgpts_veh1_dcnn_bbox, markerLength, msp4, msp1) #calculate distances in metres for DCNN method
                    
                    if(ids[j][0] == 2): #vehicle 2
                        if (detected_ID_prev[1] == 1 and diff2 < DIFF_MAX) or k == start_frame: #if this marker was detected on previous frame and its position in the image is similar
                            bbox = generatePointsBoundingBox(veh2_dim) #generate additional points for bounding box
                            if sourceLidar:
                                point = findMinimumDistanceBoundingBox(imgpts_veh4_lidar, bbox, tvec[j], rvec[j], size_corr2) #find the closest point of the bbox from Lidar
                                dist_veh2_aruco, dist_veh2_aruco_bbox = calculateDistance(imgpts_veh4_lidar, np.float32([[cx2, cy2]]), [point], markerLength, msp4, msp2) #calculate distances in metres for Aruco method
                                if drawLines:
                                    drawLinesOnImage(imgpts_veh4_lidar, point, cx2, cy2, dist_veh2_aruco, dist_veh2_aruco_bbox, ids[j][0]) #draw lines between Lidar and vehicle
                            else:
                                point = findMinimumDistanceBoundingBox(np.float32([[cx4, cy4]]), bbox, tvec[j], rvec[j], size_corr2) #find the closest point of the bbox from host's Aruco
                                dist_veh2_aruco, dist_veh2_aruco_bbox = calculateDistance(np.float32([[cx4, cy4]]), np.float32([[cx2, cy2]]), [point], markerLength, msp4, msp2) #calculate distances in metres for Aruco method
                                if drawLines:
                                    drawLinesOnImage(np.float32([[cx4, cy4]]), point, cx2, cy2, dist_veh2_aruco, dist_veh2_aruco_bbox, ids[j][0]) #draw lines between host's Aruco and vehicle
                            if useCentroidData:
                                dist_veh2_dcnn, dist_veh2_dcnn_bbox = calculateDistance(imgpts_veh4_lidar, imgpts_veh2_dcnn, imgpts_veh2_dcnn_bbox, markerLength, msp4, msp2) #calculate distances in metres for DCNN method
                    
                    if(ids[j][0] == 3): #vehicle 3
                        if (detected_ID_prev[2] == 1 and diff3 < DIFF_MAX) or k == start_frame: #if this marker was detected on previous frame and its position in the image is similar
                            bbox = generatePointsBoundingBox(veh3_dim) #generate additional points for bounding box
                            if sourceLidar:
                                point = findMinimumDistanceBoundingBox(imgpts_veh4_lidar, bbox, tvec[j], rvec[j], size_corr3) #find the closest point of the bbox from Lidar
                                dist_veh3_aruco, dist_veh3_aruco_bbox = calculateDistance(imgpts_veh4_lidar, np.float32([[cx3, cy3]]), [point], markerLength, msp4, msp3) #calculate distances in metres for Aruco method
                                if drawLines:
                                    drawLinesOnImage(imgpts_veh4_lidar, point, cx3, cy3,dist_veh3_aruco, dist_veh3_aruco_bbox, ids[j][0]) #draw lines between Lidar and vehicle
                            else:
                                point = findMinimumDistanceBoundingBox(np.float32([[cx4, cy4]]), bbox, tvec[j], rvec[j], size_corr3) #find the closest point of the bbox from host's Aruco
                                dist_veh3_aruco, dist_veh3_aruco_bbox = calculateDistance(np.float32([[cx4, cy4]]), np.float32([[cx3, cy3]]), [point], markerLength, msp4, msp3) #calculate distances in metres for Aruco method
                                if drawLines:
                                    drawLinesOnImage(np.float32([[cx4, cy4]]), point, cx3, cy3, dist_veh3_aruco, dist_veh3_aruco_bbox, ids[j][0]) #draw lines between host's Aruco and vehicle
                            if useCentroidData:
                                dist_veh3_dcnn, dist_veh3_dcnn_bbox = calculateDistance(imgpts_veh4_lidar, imgpts_veh3_dcnn, imgpts_veh3_dcnn_bbox, markerLength, msp4, msp3) #calculate distances in metres for DCNN method
                    
        detected_ID_prev = detected_ID #copy detection state of the current frame to use in the next frame

#%%====================================
#IMAGE SHOW AND DATA WRITE

    #show results on image
    if showImage:
        cv2.namedWindow("Detection result", cv2.WINDOW_NORMAL)
        cv2.imshow("Detection result", frame)
        if cv2.waitKey(cv2waitKeyVal) & 0xFF == ord('q'):
            break
    
    #save results to a file
    if saveResults:
        outputData(file)
    
    #save images to a folder
    if saveImages:
        cv2.imwrite(path_output_images + "image_%04d.png" % k, frame)
    
    #increment frame number
    k = k + step_frame
    
    #skip frames from video
    if useVideo:
        for i in range(step_frame-1):
            ret, frame = video.read()
            if ret == False:
                break

if saveResults:
    file.close()
    
if useVideo:
    video.release()

if showImage:
    cv2.destroyAllWindows()
