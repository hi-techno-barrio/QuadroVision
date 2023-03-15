import cv2
import numpy as np

# Load the video files from both cameras
video1 = cv2.VideoCapture('video1.mp4')
video2 = cv2.VideoCapture('video2.mp4')

# Create a feature detector and descriptor extractor
detector = cv2.SIFT_create()
matcher = cv2.BFMatcher()

# Initialize a list of frames and keypoints for both videos
frames1 = []
keypoints1 = []
frames2 = []
keypoints2 = []

# Loop through the frames of both videos and extract features
while True:
    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()

    if not ret1 or not ret2:
        break

    # Convert the frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors in both frames
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    # Add the frames and keypoints to the lists
    frames1.append(gray1)
    keypoints1.append(kp1)
    frames2.append(gray2)
    keypoints2.append(kp2)

# Match keypoints between adjacent frames of both videos
matches = []
for i in range(len(frames1) - 1):
    # Match keypoints between adjacent frames in video 1
    matches1 = matcher.match(des1[i], des1[i+1])
    # Match keypoints between adjacent frames in video 2
    matches2 = matcher.match(des2[i], des2[i+1])
    # Add the matches to the list
    matches.append((matches1, matches2))

# Estimate the homography between adjacent frames of both videos
homographies = []
for m in matches:
    # Get the keypoints from the matches
    src_pts1 = np.float32([keypoints1[i][m[0][j].queryIdx].pt for j in range(len(m[0]))]).reshape(-1,1,2)
    dst_pts1 = np.float32([keypoints1[i+1][m[0][j].trainIdx].pt for j in range(len(m[0]))]).reshape(-1,1,2)
    src_pts2 = np.float32([keypoints2[i][m[1][j].queryIdx].pt for j in range(len(m[1]))]).reshape(-1,1,2)
    dst_pts2 = np.float32([keypoints2[i+1][m[1][j].trainIdx].pt for j in range(len(m[1]))]).reshape(-1,1,2)
    # Estimate the homography between the frames
    H1, _ = cv2.findHomography(src_pts1, dst_pts1, cv2.RANSAC, 5.0)
    H2, _ = cv2.findHomography(src_pts2, dst_pts2, cv2.RANSAC, 5.0)
    # Add the homographies to the list
    homographies.append((H1, H2))

# Initialize the stitcher and create the panorama
stitcher = cv2.createStitcher()
pano, status = stitcher.stitch([frames1[0], frames2[0]])

# Warp and blend the remaining frames into the panorama
for i in range(len(homographies)):
    # Warp the frames using the
warp1 = cv2.warpPerspective(frames1[i+1], homographies[i][0], (pano.shape[1], pano.shape[0]))
warp2 = cv2.warpPerspective(frames2[i+1], homographies[i][1], (pano.shape[1], pano.shape[0]))
# Add the warped frames to the panorama using multi-band blending
pano = cv2.seamlessClone(warp1, pano, np.ones_like(warp1), (pano.shape[1]//2, pano.shape[0]//2), cv2.NORMAL_CLONE)
pano = cv2.seamlessClone(warp2, pano, np.ones_like(warp2), (pano.shape[1]//2, pano.shape[0]//2), cv2.NORMAL_CLONE)
cv2.imshow('Panorama', pano)
cv2.waitKey(0)
cv2.destroyAllWindows()
