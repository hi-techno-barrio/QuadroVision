# In this modified code, we add object detection and tracking to detect objects in each frame and compute the distance and degrees of the object from each camera. 
#We also add text overlays to the final panorama to display this information. Note that the `object_detection()` and `compute_distance_and_degrees()` functions 
#are not shown here and would need to be defined separately. Additionally, you may need to adjust the code to handle different types of objects and cameras, as 
#well as different lighting conditions and background environments.

import cv2
import numpy as np

# Load the video files from all four cameras
video1 = cv2.VideoCapture('video1.mp4')
video2 = cv2.VideoCapture('video2.mp4')
video3 = cv2.VideoCapture('video3.mp4')
video4 = cv2.VideoCapture('video4.mp4')

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

# Loop through the frames of all four videos and extract features
while True:
    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()
    ret3, frame3 = video3.read()
    ret4, frame4 = video4.read()

    if not ret1 or not ret2 or not ret3 or not ret4:
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

# Match keypoints between adjacent frames of all four videos
matches = []
for i in range(len(frames1) - 1):
    # Match keypoints between adjacent frames in video 1
    matches1 = matcher.match(des1[i], des1[i+1])
    # Match keypoints between adjacent frames in video 2
    matches2 = matcher.match(des2[i], des2[i+1])
    # Match keypoints between adjacent frames in video 3
    matches3 = matcher.match(des3[i], des3[i+1])
    # Match keypoints between adjacent frames in video 4
    matches4 = matcher.match(des4[i], des4[i+1])
    # Add the matches to the list
    matches.append((matches1, matches2, matches3, matches4))

# Estimate the homography between adjacent frames of all four videos
homographies = []
for m in matches:
    # Get the keypoints from the matches
    src# Get the keypoints from the matches
src_pts1 = np.float32([keypoints1[i][m[0][j].queryIdx].pt for j in range(len(m[0]))]).reshape(-1,1,2)
dst_pts1 = np.float32([keypoints1[i+1][m[0][j].trainIdx].pt for j in range(len(m[0]))]).reshape(-1,1,2)
src_pts2 = np.float32([keypoints2[i][m[1][j].queryIdx].pt for j in range(len(m[1]))]).reshape(-1,1,2)
dst_pts2 = np.float32([keypoints2[i+1][m[1][j].trainIdx].pt for j in range(len(m[1]))]).reshape(-1,1,2)
src_pts3 = np.float32([keypoints3[i][m[2][j].queryIdx].pt for j in range(len(m[2]))]).reshape(-1,1,2)
dst_pts3 = np.float32([keypoints3[i+1][m[2][j].trainIdx].pt for j in range(len(m[2]))]).reshape(-1,1,2)
src_pts4 = np.float32([keypoints4[i][m[3][j].queryIdx].pt for j in range(len(m[3]))]).reshape(-1,1,2)
dst_pts4 = np.float32([keypoints4[i+1][m[3][j].trainIdx].pt for j in range(len(m[3]))]).reshape(-1,1,2)
# Estimate the homography between the frames
H1, _ = cv2.findHomography(src_pts1, dst_pts1, cv2.RANSAC, 5.0)
H2, _ = cv2.findHomography(src_pts2, dst_pts2, cv2.RANSAC, 5.0)
H3, _ = cv2.findHomography(src_pts3, dst_pts3, cv2.RANSAC, 5.0)
H4, _ = cv2.findHomography(src_pts4, dst_pts4, cv2.RANSAC, 5.0)
# Add the homographies to the list
homographies.append((H1, H2, H3, H4))
Initialize the stitcher and create the panorama
stitcher = cv2.createStitcher()
pano, status = stitcher.stitch([frames1[0], frames2[0], frames3[0], frames4[0]])

Warp and blend the remaining frames into the panorama
for i in range(len(homographies)):
# Warp the frames using the homography matrices
warp1 = cv2.warpPerspective(frames1[i+1], homographies[i][0], (pano.shape[1], pano.shape[0]))
warp2 = cv2.warpPerspective(frames2[i+1], homographies[i][1], (pano.shape[1], pano.shape[0]))
warp3 = cv2.warpPerspective(frames3[i+1], homographies[i][2], (pano.shape[1], pano.shape[0]))
warp4 = cv2.warpPerspective(frames4[i+1], homographies[i][3], (pano.shape[1], pano.shape[0]))
# Detect objects in each warped frame
object1 = object_detection(warp1)
object2 = object_detection(warp2)
object3 = object_detection(warp3)
object4 = object_detection(warp4)# Compute the distance and degrees of each object from each camera
dist1, degrees1 = compute_distance_and_degrees(object1, detections1[i+1])
dist2, degrees2 = compute_distance_and_degrees(object2, detections2[i+1])
dist3, degrees3 = compute_distance_and_degrees(object3, detections3[i+1])
dist4, degrees4 = compute_distance_and_degrees(object4, detections4[i+1])

# Add the object information to the panorama
text1 = f'{object1}: {dist1} meters, {degrees1} degrees'
text2 = f'{object2}: {dist2} meters, {degrees2} degrees'
text3 = f'{object3}: {dist3} meters, {degrees3} degrees'
text4 = f'{object4}: {dist4} meters, {degrees4} degrees'
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


