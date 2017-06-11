#include "scan.hpp"

// ---------------------------
//  Helper Method
// ---------------------------
bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2)
{
    double i = fabs(contourArea(cv::Mat(contour1)));
    double j = fabs(contourArea(cv::Mat(contour2)));
    return (i > j);
}


// ---------------------------
//  Scan Class Public Methods
// ---------------------------
Scan::Scan(cv::Mat& input)
{
    mInput = input;
}
Scan::Scan(char *filename)
{
    mFilename = std::string(filename);
}
void Scan::Load()
{
    mInput = cv::imread(mFilename, cv::IMREAD_UNCHANGED);
    if (mInput.rows == 0 || mInput.cols == 0)
    {
        printf("ERROR: image could not open\n");
        exit(1);
    }
}
void Scan::preprocess()
{
    int rows = mInput.rows;
    int cols = mInput.cols;

    cv::cvtColor(mInput, mRGB, CV_BGR2RGB);

    cv::Mat blur;
    cv::bilateralFilter(mRGB, blur, 11, 90, 90);

    cv::cvtColor(blur, mGray, CV_RGB2GRAY);

    cv::threshold(mGray, mThresh, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
}

int Scan::detect_edges(uint8_t lo_thresh, uint8_t hi_thresh)
{
    int rows = mInput.rows;
    int cols = mInput.cols;

    cv::Mat downRGB;
    cv::Mat downGray;
    cv::Mat downThresh;

    cv::resize(mRGB, downRGB, cv::Size(cols / 2, rows / 2));
    cv::resize(mGray, downGray, cv::Size(cols / 2, rows / 2));
    cv::resize(mThresh, downThresh, cv::Size(cols / 2, rows / 2));

    cv::Mat edgesR;
    cv::Mat edgesG;
    cv::Mat edgesB;
    cv::Mat edgesGray;
    cv::Mat edgesThresh;

    std::vector<cv::Mat> rgbChannel(3);
    cv::split(downRGB, rgbChannel);

    cv::Canny(rgbChannel[0], edgesR, lo_thresh, hi_thresh);
    cv::Canny(rgbChannel[1], edgesG, lo_thresh, hi_thresh);
    cv::Canny(rgbChannel[2], edgesB, lo_thresh, hi_thresh);
    cv::Canny(downGray, edgesGray, lo_thresh, hi_thresh);
    cv::Canny(downThresh, edgesThresh, lo_thresh, hi_thresh);

    cv::Mat downEdges;

    cv::max(edgesGray, edgesThresh, downEdges);
    cv::max(downEdges, edgesR, downEdges);
    cv::max(downEdges, edgesG, downEdges);
    cv::max(downEdges, edgesB, downEdges);

    cv::Scalar mean = cv::mean(downEdges);
    cv::threshold(downEdges, downEdges, mean[0], 255, cv::THRESH_BINARY);

    cv::Mat edges;
    cv::resize(downEdges, edges, cv::Size(cols, rows));
    // edges.convertTo(mEdges, CV_8U);
    cv::threshold(edges, mEdges, 128, 255, cv::THRESH_BINARY);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5), cv::Point(0,0));
    cv::morphologyEx(mEdges, mEdges, cv::MORPH_CLOSE, kernel);

    return 0;
}

int Scan::detect_contours()
{
    if (mEdges.rows == 0 || mEdges.cols == 0)
        this->detect_edges();

    int rows = mInput.rows;
    int cols = mInput.cols;

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(mEdges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::sort(contours.begin(), contours.end(), compareContourAreas);

    for (int i = 0; i < contours.size(); i++)
    {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contours[i], approx, 0.004 * cv::arcLength(contours[i], true), true);

        if (cv::contourArea(approx) < rows * cols * 0.005)
            continue;

        contours_pts.push_back(approx);
        contours_boxes.push_back(cv::boundingRect(approx));

        cv::Moments m = cv::moments(approx);
        cv::Point centroid;
        centroid.x = m.m10 / m.m00;
        centroid.y = m.m01 / m.m00;
        contours_centroids.push_back(centroid);
    }

    if (contours.size() == 0)
    {
        printf("object detected has fallen out of the camera capture region or is not visible."
               "Current process is being stopped and all the data for current scan will be lost."
               "Please re-align the object and restart the scanning\n");
        exit(1);
    }

    return 0;

}
int Scan::Video()
{
    cv::VideoCapture inputVideo(mFilename);
    if(!inputVideo.isOpened())
    {
        printf ("can not open video file\n");
        return -1;
    }
    std::string::size_type pAt = mFilename.find_last_of('.');  // Find extension point
    const std::string outvid = mFilename.substr(0, pAt) + "_OUT.mp4";   // Form the new name with container
    int ex = static_cast<int>(inputVideo.get(CV_CAP_PROP_FOURCC));     // Get Codec Type- Int form

    // Transform from int to char via Bitwise operators
    char EXT[] = {(char)(ex & 0XFF) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0};

    cv::Size S = cv::Size((int) inputVideo.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));

    cv::VideoWriter outputVideo; // Open the output
    printf("outvid: %s\n", outvid.c_str());
    outputVideo.open(outvid, ex, inputVideo.get(CV_CAP_PROP_FPS), S, true);

    if (!outputVideo.isOpened())
    {
        printf("Could not open the output video for write: %s\n", mFilename.c_str());
        return -1;
    }

    for (int i = 0; ; i++)
    {
        inputVideo >> mInput;
        if (mInput.empty())
            break;

        contours_pts.clear();
        contours_boxes.clear();
        contours_centroids.clear();
        preprocess();
        detect_edges();
        detect_contours();
        draw();

        outputVideo << vis_in;

        display();
        char key = cv::waitKey(10);
        if (key == 27)
            break;
    }

    return 0;
}

void Scan::draw()
{
    vis_in = mInput.clone();

    int rows = mInput.rows;
    int cols = mInput.cols;

    cv::RNG rng(12345);

    for (int i = 0; i < contours_pts.size(); i++)
    {
        cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(150,255));
        cv::drawContours(vis_in, contours_pts, i, color, 2, 8, std::vector<cv::Vec4i>(), 0, cv::Point());

        // cv::Rect bbox = contours_boxes[i];
        // if ((bbox.x + bbox.width >= cols) ||
        //     (bbox.y + bbox.height >= rows) ||
        //     (bbox.x <= 0) ||
        //     (bbox.y <= 0))
        //     cv::rectangle(vis_in, contours_boxes[i].tl(), contours_boxes[i].br(), cv::Scalar(255, 0, 0), 2, 8, 0);
        // else
        //     cv::rectangle(vis_in, contours_boxes[i].tl(), contours_boxes[i].br(), color, 2, 8, 0);

        cv::circle(vis_in, contours_centroids[i], 4, color, -1);
    }

    cv::circle(vis_in, cv::Point(cols/2, rows/2), 2, cv::Scalar(0,0,255),-1);
}
void Scan::display()
{
    cv::imshow("vis_in" , vis_in);
    cv::imshow("Input"  , mInput);
    cv::imshow("Thresh" , mThresh);
    cv::imshow("Gray"   , mGray);
    cv::imshow("Edges"  , mEdges);
}