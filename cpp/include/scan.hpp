
#ifndef __SCAN_HPP__
#define __SCAN_HPP__

#include <iostream>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class Scan
{
    private:
        std::string mFilename;

        cv::Mat mGray;
        cv::Mat mRGB;
        cv::Mat mEdges;
        cv::Mat mInput;
        cv::Mat mThresh;
        cv::Mat mCropped;

        cv::Mat vis_in;

        std::vector<cv::Rect>                   contours_boxes;
        std::vector<cv::Point>                  contours_centroids;
        std::vector<std::vector<cv::Point> >    contours_pts;

    public:
        Scan(char *filename);
        void Load();
        void preprocess();
        int detect_contours();
        int detect_edges(uint8_t lo_thresh=80, uint8_t hi_thresh=160);

        int Video();
        void draw();
        void display();
};


#endif