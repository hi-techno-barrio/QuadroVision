#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    // Load the video files from all four cameras
    VideoCapture video1("video1.mp4");
    VideoCapture video2("video2.mp4");
    VideoCapture video3("video3.mp4");
    VideoCapture video4("video4.mp4");

    // Check if videos are successfully opened
    if (!video1.isOpened() || !video2.isOpened() || !video3.isOpened() || !video4.isOpened()) {
        cout << "Error opening video files!" << endl;
        return -1;
    }

    // Create a feature detector and descriptor extractor
    Ptr<SIFT> detector = SIFT::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");

    // Initialize a list of frames, keypoints, and object detections for all four videos
    vector<Mat> frames1, frames2, frames3, frames4;
    vector<vector<KeyPoint>> keypoints1, keypoints2, keypoints3, keypoints4;
    vector<Rect> detections1, detections2, detections3, detections4;

    // Loop through the frames of all four videos and extract features
    while (true)
    {
        Mat frame1, frame2, frame3, frame4;
        bool success1 = video1.read(frame1);
        bool success2 = video2.read(frame2);
        bool success3 = video3.read(frame3);
        bool success4 = video4.read(frame4);

        if (!success1 || !success2 || !success3 || !success4) {
            break;
        }

        // Convert the frames to grayscale
        Mat gray1, gray2, gray3, gray4;
        cvtColor(frame1, gray1, COLOR_BGR2GRAY);
        cvtColor(frame2, gray2, COLOR_BGR2GRAY);
        cvtColor(frame3, gray3, COLOR_BGR2GRAY);
        cvtColor(frame4, gray4, COLOR_BGR2GRAY);

        // Detect keypoints and descriptors in all four frames
        vector<KeyPoint> kp1, kp2, kp3, kp4;
        detector->detect(gray1, kp1);
        detector->detect(gray2, kp2);
        detector->detect(gray3, kp3);
        detector->detect(gray4, kp4);

        Mat des1, des2, des3, des4;
        detector->compute(gray1, kp1, des1);
        detector->compute(gray2, kp2, des2);
        detector->compute(gray3, kp3, des3);
        detector->compute(gray4, kp4, des4);

        // Add the frames and keypoints to the lists
        frames1.push_back(gray1);
        frames2.push_back(gray2);
        frames3.push_back(gray3);
        frames4.push_back(gray4);
        keypoints1.push_back(kp1);
        keypoints2.push_back(kp2);
        keypoints3.push_back(kp3);
        keypoints4.push_back(kp4);

        // Match the descriptors to find object detections in all four frames
        vector<vector<DMatch>> matches1, matches2, matches3, matches4;
        matcher->knnMatch(des1, des1, matches1, 2);
        matcher->knnMatch(des2, des2, matches2, 2);
        matcher->knnMatch(des3, des3, matches3, 2);
        matcher->knnMatch(des4, des4, matches4, 2);

// Add the final warped frames to the panorama using multi-band blending
cvSeamlessClone(warp1, pano, mask1, cvPoint(pano->width/2, pano->height/2), panorama, CV_NORMAL_CLONE);
cvSeamlessClone(warp2, pano, mask2, cvPoint(pano->width/2, pano->height/2), panorama, CV_NORMAL_CLONE);
cvSeamlessClone(warp3, pano, mask3, cvPoint(pano->width/2, pano->height/2), panorama, CV_NORMAL_CLONE);
cvSeamlessClone(warp4, pano, mask4, cvPoint(pano->width/2, pano->height/2), panorama, CV_NORMAL_CLONE);

// Display the final panorama with object information
cvPutText(panorama, text1, cvPoint(50, 50), &font, cvScalar(0, 0, 255, 0));
cvPutText(panorama, text2, cvPoint(50, 100), &font, cvScalar(0, 0, 255, 0));
cvPutText(panorama, text3, cvPoint(50, 150), &font, cvScalar(0, 0, 255, 0));
cvPutText(panorama, text4, cvPoint(50, 200), &font, cvScalar(0, 0, 255, 0));
cvShowImage("Panorama", panorama);
cvWaitKey(0);

// Release memory
cvReleaseImage(&frame1);
cvReleaseImage(&frame2);
cvReleaseImage(&frame3);
cvReleaseImage(&frame4);
cvReleaseImage(&gray1);
cvReleaseImage(&gray2);
cvReleaseImage(&gray3);
cvReleaseImage(&gray4);
cvReleaseImage(&pano);
cvReleaseImage(&warp1);
cvReleaseImage(&warp2);
cvReleaseImage(&warp3);
cvReleaseImage(&warp4);
cvReleaseImage(&mask1);
cvReleaseImage(&mask2);
cvReleaseImage(&mask3);
cvReleaseImage(&mask4);
cvReleaseImage(&object_mask1);
cvReleaseImage(&object_mask2);
cvReleaseImage(&object_mask3);
cvReleaseImage(&object_mask4);

// Release videos
cvReleaseCapture(&video1);
cvReleaseCapture(&video2);
cvReleaseCapture(&video3);
cvReleaseCapture(&video4);

return 0;
}

void object_detection(IplImage* frame, char* name, float* distance, float* angle)
{
    // Load the pre-trained classifier for object detection
    CvHaarClassifierCascade* cascade = (CvHaarClassifierCascade*) cvLoad("haarcascade_frontalface_alt.xml");

    // Convert the image to grayscale
    IplImage* gray = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
    cvCvtColor(frame, gray, CV_BGR2GRAY);

    // Detect objects in the image
    CvSeq* objects = cvHaarDetectObjects(gray, cascade, cvCreateMemStorage(), 1.1, 3, CV_HAAR_DO_CANNY_PRUNING, cvSize(30, 30));

    // Check if an object was detected
    if (objects->total > 0) {
        // Get the first object detected
        CvRect* r = (CvRect*) cvGetSeqElem(objects, 0);

        // Compute the center of the object
        int x = r->x + r->width/2;
        int y = r->y + r->height/2;

        // Compute the distance and angle of the object from the center of the image
        *distance = sqrt(pow(x - frame->width/2, 2) + pow(y - frame->height/2, 2));
        *angle = atan2(y - frame->height/2, x - frame->width/2) * 180 / CV_PI;

        // Set the name of the detected object
        strcpy(name, "Person");
    }
    else {
        // No object was detected
        strcpy(name, "None");
        *distance = 0;
        *angle = 0;
    }

    // Release memory
    cvReleaseImage(&gray);
    cvReleaseHaarClassifierCascade(&cascade);
}

// Function to calculate the distance and angle of an object in an image
// Returns the distance and angle in degrees
void calculate_distance_and_angle(IplImage* frame, float* distance, float* angle)
{
    // Convert the image to grayscale
    IplImage* gray = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
    cvCvtColor(frame, gray, CV_BGR2GRAY);

    // Detect edges using Canny edge detector
    IplImage* edges = cvCreateImage(cvGetSize(gray), IPL_DEPTH_8U, 1);
    cvCanny(gray, edges, 100, 200);

    // Find contours in the image
    CvSeq* contours;
    CvMemStorage* storage = cvCreateMemStorage(0);
    cvFindContours(edges, storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

    // Find the largest contour in the image
    CvSeq* largest_contour = NULL;
    double largest_area = 0.0;
    for (CvSeq* c = contours; c != NULL; c = c->h_next) {
        double area = fabs(cvContourArea(c));
        if (area > largest_area) {
            largest_contour = c;
            largest_area = area;
        }
    }

    // Find the center of mass of the largest contour
    CvMoments moments;
    cvMoments(largest_contour, &moments);
    double cx = moments.m10 / moments.m00;
    double cy = moments.m01 / moments.m00;

    // Calculate the distance and angle of the object
    double image_width = frame->width;
    double image_height = frame->height;
    double fov = 60.0; // Camera field of view in degrees
    double aspect_ratio = image_width / image_height;
    double focal_length = image_width / (2.0 * tan(fov * M_PI / 360.0));
    double x = cx - image_width / 2.0;
    double y = cy - image_height / 2.0;
    double object_distance = sqrt(pow(x, 2.0) + pow(y, 2.0)) * (1.0 / focal_length);
    double object_angle = atan(x / (focal_length * aspect_ratio)) * 180.0 / M_PI;

    // Return the distance and angle of the object
    *distance = object_distance;
    *angle = object_angle;

    // Cleanup
    cvReleaseMemStorage(&storage);
    cvReleaseImage(&gray);
    cvReleaseImage(&edges);
}


char* calculate_orientation(float roll, float pitch, float yaw)
{
    // Convert roll, pitch, and yaw angles from degrees to radians
    float phi = roll * CV_PI / 180.0;
    float theta = pitch * CV_PI / 180.0;
    float psi = yaw * CV_PI / 180.0;

    // Compute the rotation matrix
    float R11 = cos(theta) * cos(psi);
    float R12 = -cos(phi) * sin(psi) + sin(phi) * sin(theta) * cos(psi);
    float R13 = sin(phi) * sin(psi) + cos(phi) * sin(theta) * cos(psi);
    float R21 = cos(theta) * sin(psi);
    float R22 = cos(phi) * cos(psi) + sin(phi) * sin(theta) * sin(psi);
    float R23 = -sin(phi) * cos(psi) + cos(phi) * sin(theta) * sin(psi);
    float R31 = -sin(theta);
    float R32 = sin(phi) * cos(theta);
    float R33 = cos(phi) * cos(theta);

    // Compute the XYZ orientation of the detected object
    float x = R31;
    float y = R32;
    float z = R33;

    // Create the orientation string
    static char orientation[50];
    sprintf(orientation, "X: %0.2f Y: %0.2f Z: %0.2f", x, y, z);

    return orientation;
}


// Function to get GPS coordinates
// Returns the latitude and longitude as a string
char* get_gps_coordinates()
{
    // Initialize the GPS data structures
    gps_init();
    struct timeval tv;
    tv.tv_sec = 5;
    tv.tv_usec = 0;

    // Wait for a fix
    while (gps_waiting(&gps_data, 5000)) {
        if (gps_read(&gps_data) == -1) {
            continue;
        }
        if (gps_data.status == STATUS_FIX) {
            break;
        }
    }

    // Get the latitude and longitude
    char* coordinates = (char*)malloc(100*sizeof(char));
    sprintf(coordinates, "Latitude: %f Longitude: %f", gps_data.fix.latitude, gps_data.fix.longitude);
    return coordinates;
}

// Combine the four warped frames into a single panoramic video
cvSeamlessClone(warp1, pano, mask1, cvPoint(pano->width/2, pano->height/2), panorama, CV_NORMAL_CLONE);
cvSeamlessClone(warp2, pano, mask2, cvPoint(pano->width/2, pano->height/2), panorama, CV_NORMAL_CLONE);
cvSeamlessClone(warp3, pano, mask3, cvPoint(pano->width/2, pano->height/2), panorama, CV_NORMAL_CLONE);
cvSeamlessClone(warp4, pano, mask4, cvPoint(pano->width/2, pano->height/2), panorama, CV_NORMAL_CLONE);

// Detect objects in the final panorama
object_detection(panorama);

// Display the final panorama
cvNamedWindow("Panorama", CV_WINDOW_AUTOSIZE);
cvShowImage("Panorama", panorama);
cvWaitKey(0);

// Release all resources
cvReleaseImage(&frame1);
cvReleaseImage(&frame2);
cvReleaseImage(&frame3);
cvReleaseImage(&frame4);
cvReleaseImage(&gray1);
cvReleaseImage(&gray2);
cvReleaseImage(&gray3);
cvReleaseImage(&gray4);
cvReleaseImage(&mask1);
cvReleaseImage(&mask2);
cvReleaseImage(&mask3);
cvReleaseImage(&mask4);
cvReleaseImage(&warp1);
cvReleaseImage(&warp2);
cvReleaseImage(&warp3);
cvReleaseImage(&warp4);
cvReleaseImage(&pano);
cvReleaseImage(&panorama);

return 0;
}

// Function to detect objects in a frame and add text overlays to the final panorama
void object_detection(IplImage* panorama) {
// Load the Haar classifier
CvHaarClassifierCascade* classifier = (CvHaarClassifierCascade*)cvLoad("haarcascade_frontalface_default.xml", 0, 0, 0);  // Set the detection parameters
CvMemStorage* storage = cvCreateMemStorage(0);
CvSeq* faces = cvHaarDetectObjects(panorama, classifier, storage, 1.2, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(30, 30));

// Loop through each detected object and add text overlays to the final panorama
for (int i = 0; i < faces->total; i++) {
    CvRect* face = (CvRect*)cvGetSeqElem(faces, i);
    float dist = calculate_distance(face->width);
    float degrees = calculate_degrees(panorama->width, face->x);
    char text[100];
    sprintf(text, "Face: %0.2f meters, %0.2f degrees", dist, degrees);
    CvFont font;
    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 2, CV_AA);
    cvPutText(panorama, text, cvPoint(face->x, face->y - 20), &font, cvScalar(0, 0, 255), 2, CV_AA);
}

// Release the resources
cvReleaseHaarClassifierCascade(&classifier);
cvReleaseMemStorage(&storage);
}

// Function to calculate the distance to an object based on its width in the frame
float calculate_distance(int width) {
float focal_length = 500.0; // Focal length of the camera (in pixels)
float real_width =    // Add the text overlays to the final panorama
    char text1[50], text2[50], text3[50], text4[50];
    sprintf(text1, "%s: %0.2f meters, %0.2f degrees, X:%0.2f Y:%0.2f Z:%0.2f", object_name1, dist1, degrees1, orientation1.x, orientation1.y, orientation1.z);
    sprintf(text2, "%s: %0.2f meters, %0.2f degrees, X:%0.2f Y:%0.2f Z:%0.2f", object_name2, dist2, degrees2, orientation2.x, orientation2.y, orientation2.z);
    sprintf(text3, "%s: %0.2f meters, %0.2f degrees, X:%0.2f Y:%0.2f Z:%0.2f", object_name3, dist3, degrees3, orientation3.x, orientation3.y, orientation3.z);
    sprintf(text4, "%s: %0.2f meters, %0.2f degrees, X:%0.2f Y:%0.2f Z:%0.2f", object_name4, dist4, degrees4, orientation4.x, orientation4.y, orientation4.z);

    CvFont font;
    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 2, CV_AA);

    cvPutText(panorama, text1, cvPoint(50, 50), &font, cvScalar(0, 0, 255, 0));
    cvPutText(panorama, text2, cvPoint(50, 100), &font, cvScalar(0, 0, 255, 0));
    cvPutText(panorama, text3, cvPoint(50, 150), &font, cvScalar(0, 0, 255, 0));
    cvPutText(panorama, text4, cvPoint(50, 200), &font, cvScalar(0, 0, 255, 0));

    // Show the final panorama
    cvNamedWindow("Panorama", CV_WINDOW_NORMAL);
    cvResizeWindow("Panorama", 800, 600);
    cvShowImage("Panorama", panorama);

    cvWaitKey(0);

    // Clean up
    cvReleaseImage(&frame1);
    cvReleaseImage(&frame2);
    cvReleaseImage(&frame3);
    cvReleaseImage(&frame4);
    cvReleaseImage(&gray1);
    cvReleaseImage(&gray2);
    cvReleaseImage(&gray3);
    cvReleaseImage(&gray4);
    cvReleaseImage(&panorama);
    cvReleaseImage(&warp1);
    cvReleaseImage(&warp2);
    cvReleaseImage(&warp3);
    cvReleaseImage(&warp4);
    cvReleaseImage(&mask1);
    cvReleaseImage(&mask2);
    cvReleaseImage(&mask3);
    cvReleaseImage(&mask4);

    return 0;
}
