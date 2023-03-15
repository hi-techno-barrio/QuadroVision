# QuadroVision-360.c


Design 1
Loads four video files: The code loads four video files from the file system, one for each camera position.

Extracts features and descriptors: The code uses the SIFT feature detector to extract key points and descriptors from each frame of all four videos.

Matches corresponding key points: The code uses the brute force matcher to match corresponding key points between adjacent frames of all four videos.

Estimates homography: The code uses the findHomography() function to estimate the homography between adjacent frames of all four videos. Since the camera positions are 90 degrees apart, additional information is passed to findHomography() to handle the perspective distortion between the camera views.

Warps and blends frames: The code uses the warpPerspective() function to warp the frames using the homography matrices, and the seamlessClone() function to blend the warped frames into the final panoramic view using multi-band blending.

Displays the final panorama: The code uses OpenCV's imshow() function to display the final panorama.

Uses a loop to process all frames: The code uses a loop to process all frames of all four videos, extracting features, matching key points, estimating homography, and warping and blending frames.

Is modular: The code is modular, with each step of the process encapsulated in its own function or loop.

Is customizable: The code can be easily customized to handle different numbers of cameras, camera positions, and video file formats, as well as different feature detection and matching algorithms, homography estimation methods, and blending techniques.

Overall, this code demonstrates how to use OpenCV's stitching module to stitch together video streams from multiple cameras into a single panoramic video

Design 2
Wider field of view: Using multiple cameras positioned 90 degrees apart provides a wider field of view compared to traditional single-camera setups.
Higher resolution: Combining the images from multiple cameras can result in a higher resolution panoramic image compared to stitching together images from a single camera.
Object tracking and orientation: By integrating object detection and an IMU, the code can track objects and provide their distance, degrees, and orientation relative to the camera.
Real-time display: The code can display the panoramic video in real-time, allowing for immediate feedback and monitoring.
Open-source: The code is open-source and can be easily modified and customized to meet specific requirements.

Design 3:
The features of the program are:

Four cameras positioned at 90-degree intervals covering 360 degrees to produce a single panoramic video.

Object detection using OpenCV library to detect objects in each camera's field of view.

Distance and degrees calculations to determine the location of the detected object from each camera.

Use of MPU6050 sensor data to calculate the orientation of the camera.

Use of GPS coordinates to determine the exact location of the detected object.

Calculation of the XYZ orientation of the detected object from each camera.

Text overlays added to the final panorama to display object name, distance, degrees, and orientation.

Multi-band blending to combine the four warped frames into a final panoramic video.

Real-time processing of video streams.

Overall, the program uses multiple sensors and libraries to detect objects, determine their location and orientation, and combine the video streams into a panoramic view with object information displayed. This has potential applications in surveillance, security, and other areas where it is important to monitor a wide area and detect objects with accuracy.
