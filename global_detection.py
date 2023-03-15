import cv2
import numpy as np
from mpu6050 import mpu6050
from geopy.distance import distance as gps_distance

# Initialize the MPU6050
mpu = mpu6050(0x68)

# Load the video files from all four cameras
video1 = cv2.VideoCapture('video1.mp4')
video2 = cv2.VideoCapture('video2.mp4')
video3 = cv2.VideoCapture('video3.mp4')
video4 = cv2.VideoCapture('video4.mp4')

# Set the GPS coordinates of the user's position
user_lat, user_lon = 37.7749, -122.4194

# Create a feature detector and descriptor extractor
detector = cv2.SIFT_create()
matcher = cv2.BFMatcher()

# Initialize a list of frames, keypoints, and object detections for all four videos
frames1 = []
keypoints1 = []
detections1 = []
frames2 = []
keypoints2 = []
detections2 = []
frames3 = []
keypoints3 = []
detections3 = []
frames4 = []
keypoints4 = []
detections4 = []

# Define a function to calculate the roll, pitch, and yaw angles of the camera from the MPU6050 data
def calculate_orientation(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z):
    accel_roll = np.arctan2(accel_y, accel_z)
    accel_pitch = np.arctan2(-accel_x, np.sqrt(accel_y ** 2 + accel_z ** 2))
    gyro_roll = np.deg2rad(gyro_x) * 0.95 + np.deg2rad(np.arctan2(-accel_x, np.sqrt(accel_y ** 2 + accel_z ** 2))) * 0.05
    gyro_pitch = np.deg2rad(gyro_y) * 0.95 + np.deg2rad(np.arctan2(accel_y, accel_z)) * 0.05
    yaw = np.deg2rad(gyro_z)
    roll = np.rad2deg(gyro_roll)
    pitch = np.rad2deg(gyro_pitch)
    return roll, pitch, yaw

# Define a function to compute the XYZ orientation of the detected object
def compute_orientation(roll, pitch, yaw, detection, object_lat, object_lon):
    # Convert degrees to radians
    roll_rad = np.deg2rad(roll)
    pitch_rad = np.deg2rad(pitch)
    yaw_rad = np.deg2rad(yaw)

    # Get the object's pixel coordinates
    x, y, w, h = detection
    object_x = x + w/2
    object_y = y + h/2

    # Compute the distance between the user's position and the object
    object_distance = gps_distance((user_lat, user_lon), (object_lat, object_lon)).m

    # Compute the angle between the user's position and the object
    object_angle = np.rad2deg(np.arctan2(object_x - 960, 1080))

    # Compute the XYZ orientation of the object
    x = object_distance * np.cos(roll_rad) * np.cos(yaw_rad) + object_distance * np.sin(roll_rad) * np.sin(pitch_rad) * np.sin(yaw_rad)
    y = -object_distance * np.cos(pitch_rad) * np.sin(yaw_rad) + object_distance * np.sin(roll_rad) * np.sin(pitch_rad) * np.cos
(yaw_rad)
z = object_distance * np.sin(pitch_rad) - object_distance * np.sin(roll_rad) * np.cos(pitch_rad) * np.cos(yaw_rad)
return f'{x:.2f}, {y:.2f}, {z:.2f}'
Loop through the frames of all four videos and extract features
while True:
ret1, frame1 = video1.read()
ret2, frame2 = video2.read()
ret3, frame3 = video3.read()
ret4, frame4 = video4.read()if not ret1 or not ret2 or not ret3 or not ret4:
    break

# Convert the frames to grayscale
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
gray4 = cv2.cvtColor(frame4, cv2.COLOR_BGR2GRAY)

# Detect keypoints and descriptors in all four frames
kp1, des1 = detector.detectAndCompute(gray1, None)
kp2, des2 = detector.detectAndCompute(gray2, None)
kp3, des3 = detector.detectAndCompute(gray3, None)
kp4, des4 = detector.detectAndCompute(gray4, None)

# Add the frames, keypoints, and object detections to the lists
frames1.append(gray1)
keypoints1.append(kp1)
frames2.append(gray2)
keypoints2.append(kp2)
frames3.append(gray3)
keypoints3.append(kp3)
frames4.append(gray4)
keypoints4.append(kp4)
detections1.append(object_detection(frame1))
detections2.append(object_detection(frame2))
detections3.append(object_detection(frame3))
detections4.append(object_detection(frame4))

# Get the orientation of the camera from the MPU6050
accel_data = mpu.get_accel_data()
gyro_data = mpu.get_gyro_data()
roll, pitch, yaw = calculate_orientation(accel_data['x'], accel_data['y'], accel_data['z'], gyro_data['x'], gyro_data['y'], gyro_data['z'])

# Compute the XYZ orientation of the detected object from each camera
object1 = detections1[-1][0]
object2 = detections2[-1][0]
object3 = detections3[-1][0]
object4 = detections4[-1][0]
dist1 = detections1[-1][1]
dist2 = detections2[-1][1]
dist3 = detections3[-1][1]
dist4 = detections4[-1][1]
degrees1 = object_angle(object1, frame1.shape[1])
degrees2 = object_angle(object2, frame2.shape[1])
degrees3 = object_angle(object3, frame3.shape[1])
degrees4 = object_angle(object4, frame4.shape[1])
orientation1 = compute_orientation(roll, pitch, yaw, detections1[-1], object1.lat, object1.lon)
orientation2 = compute_orientation(roll, pitch, yaw, detections2[-1], object2.lat, object2.lon)
orientation3 = compute_orientation(roll, pitch, yaw, detections3[-1], object3.lat, object3.lon)
orientation4 = compute_orientation(roll, pitch, yaw, detections4[-1], object4.lat, object4.lon)

# Add the orientation information to the panorama
text1 = f'{object1}:
dist1} meters, {degrees1} degrees, {orientation1}'
text2 = f'{object2}: {dist2} meters, {degrees2} degrees, {orientation2}'
text3 = f'{object3}: {dist3} meters, {degrees3} degrees, {orientation3}'
text4 = f'{object4}: {dist4} meters, {degrees4} degrees, {orientation4}'
cv2.putText(warp1, text1, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(warp2, text2, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(warp3, text3, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(warp4, text4, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Add the warped frames to the panorama using multi-band blending
pano = cv2.seamlessClone(warp1, pano, np.ones_like(warp1), (pano.shape[1]//2, pano.shape[0]//2), cv2.NORMAL_CLONE)
pano = cv2.seamlessClone(warp2, pano, np.ones_like(warp2), (pano.shape[1]//2, pano.shape[0]//2), cv2.NORMAL_CLONE)
pano = cv2.seamlessClone(warp3, pano, np.ones_like(warp3), (pano.shape[1]//2, pano.shape[0]//2), cv2.NORMAL_CLONE)
pano = cv2.seamlessClone(warp4, pano, np.ones_like(warp4), (pano.shape[1]//2, pano.shape[0]//2), cv2.NORMAL_CLONE)

# Display the final panorama
cv2.imshow('Panorama', pano)
cv2.waitKey(0)
cv2.destroyAllWindows()
