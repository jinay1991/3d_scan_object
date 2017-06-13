
#ifndef __SCAN_HPP__
#define __SCAN_HPP__

#include <iostream>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

struct contours_info
{
    std::vector<cv::Rect>                   max_boxes;
    std::vector<cv::Rect>                   boxes;
    std::vector<cv::Point>                  centroids;
    std::vector<std::vector<cv::Point> >    pts;
    std::vector<bool>                       alignH;
};

class Scan
{
    private:
        std::string mFilename;
        int thresh_val;

        cv::Mat mGray;
        cv::Mat mRGB;
        cv::Mat mEdges;
        cv::Mat mInput;
        cv::Mat mThresh;
        cv::Mat mCropped;

        struct contours_info contours_info;

        /* cummulative informations */
        std::vector<struct contours_info> instance;
        std::vector<cv::Mat> outFrames;

        void InitParams();

    public:
        cv::Mat vis_in;

        Scan(cv::Mat& input);
        Scan(char *filename);
        void LoadImage();
        void preprocess();
        int detect_contours();
        int detect_edges(uint8_t lo_thresh=80, uint8_t hi_thresh=160);
        void realign();
        int Video();
        int Image(bool debug=false);
        void draw();
        void display();
};


#endif