#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

#define FRAME_WIDTH 640
#define FRAME_HEIGHT 480

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    // create capture objects for the four cameras
    VideoCapture cap1(0);
    VideoCapture cap2(1);
    VideoCapture cap3(2);
    VideoCapture cap4(3);

    if (!cap1.isOpened() || !cap2.isOpened() || !cap3.isOpened() || !cap4.isOpened()) {
        printf("Error opening cameras\n");
        return -1;
    }

    // set the resolution for each camera
    cap1.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    cap1.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
    cap2.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    cap2.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
    cap3.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    cap3.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
    cap4.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    cap4.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);

    // create a stitcher object to stitch the frames together
    Ptr<Stitcher> stitcher = Stitcher::createDefault();

    while (1) {
        // read frames from the four cameras
        Mat frame1, frame2, frame3, frame4;
        cap1.read(frame1);
        cap2.read(frame2);
        cap3.read(frame3);
        cap4.read(frame4);

        // create a list of frames to stitch
        vector<Mat> frames;
        frames.push_back(frame1);
        frames.push_back(frame2);
        frames.push_back(frame3);
        frames.push_back(frame4);

        // stitch the frames together to create a panorama
        Mat pano;
        Stitcher::Status status = stitcher->stitch(frames, pano);

        if (status == Stitcher::OK) {
            // display the panorama
            imshow("panorama", pano);
        } else {
            printf("Error stitching images: %d\n", status);
        }

        // wait for key press
        if (waitKey(1) == 27) {
            break;
        }
    }

    // release the capture objects
    cap1.release();
    cap2.release();
    cap3.release();
    cap4.release();

    return 0;
}
