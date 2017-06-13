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
    InitParams();
    instance.clear();
}
Scan::Scan(char *filename)
{
    mFilename = std::string(filename);
    InitParams();
    instance.clear();
}
void Scan::InitParams()
{
    thresh_val = 80;
    contours_info.pts.clear();
    contours_info.boxes.clear();
    contours_info.centroids.clear();
    contours_info.max_boxes.clear();
    contours_info.alignH.clear();
}
void Scan::LoadImage()
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
    cv::cvtColor(mInput, mRGB, CV_BGR2RGB);

    cv::Mat blur, blur1;
    // cv::GaussianBlur(mRGB, blur1, cv::Size(9, 9), 0);
    cv::bilateralFilter(mRGB, blur, 11, 90, 90);

    cv::cvtColor(blur, mGray, CV_RGB2GRAY);

    cv::threshold(mGray, mThresh, thresh_val, 255, cv::THRESH_BINARY_INV);
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

    cv::Canny(rgbChannel[0], edgesR, lo_thresh, hi_thresh, 3);
    cv::Canny(rgbChannel[1], edgesG, lo_thresh, hi_thresh, 3);
    cv::Canny(rgbChannel[2], edgesB, lo_thresh, hi_thresh, 3);
    cv::Canny(downGray, edgesGray, lo_thresh, hi_thresh, 3);
    cv::Canny(downThresh, edgesThresh, lo_thresh, hi_thresh, 3);

    cv::Mat downEdges;

    cv::max(edgesGray, edgesThresh, downEdges);
    cv::max(downEdges, edgesR, downEdges);
    cv::max(downEdges, edgesG, downEdges);
    cv::max(downEdges, edgesB, downEdges);

    cv::Scalar mean = cv::mean(downEdges);
    cv::threshold(downEdges, downEdges, mean[0], 255, cv::THRESH_BINARY);

    cv::Mat edges;
    cv::resize(downEdges, edges, cv::Size(cols, rows));
    cv::threshold(edges, mEdges, 128, 255, cv::THRESH_BINARY);
    cv::bitwise_and(mEdges, mEdges, mEdges, mThresh);

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

    bool alignH = false;
    int Right = 0;
    int Left = cols;
    int Bottom = 0;
    int Top = rows;
    for (int i = 0; i < contours.size(); i++)
    {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contours[i], approx, 0.004 * cv::arcLength(contours[i], true), true);

        if (cv::contourArea(approx) < rows * cols * 0.005)
            continue;

        contours_info.pts.push_back(approx);

        cv::Rect bbox = cv::boundingRect(approx);
        contours_info.boxes.push_back(bbox);

        cv::Moments m = cv::moments(approx);
        cv::Point centroid;
        centroid.x = m.m10 / m.m00;
        centroid.y = m.m01 / m.m00;
        contours_info.centroids.push_back(centroid);

        if ((centroid.y > cols * 0.70 || centroid.y < cols * 0.30) && !alignH)
             alignH = true;
        contours_info.alignH.push_back(alignH);

        Right = std::max(bbox.x, Right);
        Left = std::min(bbox.x, Left);
        Bottom = std::max(bbox.y, Bottom);
        Top = std::min(bbox.y, Top);
        int width = Right - Left;
        int height = Bottom - Top;
        cv::Rect max_box;
        max_box.x=Left;
        max_box.y=Top;
        max_box.width=width;
        max_box.height=height;

        contours_info.max_boxes.push_back(max_box);
    }

    if (contours_info.pts.size() == 0)
    {
        printf("object detected has fallen out of the camera capture region or is not visible."
               "Current process is being stopped and all the data for current scan will be lost."
               "Please re-align the object and restart the scanning\n");
        exit(1);
    }

    return 0;

}

void Scan::realign()
{
    int rows = mInput.rows;
    int cols = mInput.cols;

    cv::Mat wrap_out = mRGB.clone();

    for (int i = 0; i < contours_info.pts.size(); i++)
    {
        std::vector<cv::Point> cnt = contours_info.pts[i];

        // get centroid
        cv::Point centroid = contours_info.centroids[i];

        int x_disp = (cols / 2) - centroid.x;
        int y_disp = contours_info.alignH[i] ? (rows / 2) - centroid.y : 0;

        cv::Mat M = (cv::Mat_<double>(2,3) << 1, 0, x_disp, 0, 1, y_disp);
        cv::warpAffine(mRGB, wrap_out, M, cv::Size(cols, rows), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    }

    InitParams();
    cv::cvtColor(wrap_out, mInput, CV_RGB2BGR);
    preprocess();
    detect_edges();
    detect_contours();
}


// ----------------------------------
//  APIs
// ----------------------------------
int Scan::Image(bool debug)
{
    preprocess();
    detect_edges();
    detect_contours();
    realign();
    draw();
    if (debug)
    {
        display();
        cv::waitKey(0);
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

    printf("outvid: %s\n", outvid.c_str());
    

    for (int i = 0; ; i++)
    {
        inputVideo >> mInput;
        if (mInput.empty())
            break;

        InitParams();
        preprocess();
        detect_edges();
        detect_contours();
        realign();
        draw();

        instance.push_back(contours_info);
        outFrames.push_back(vis_in.clone());

        display();
        char key = cv::waitKey(10);
        if (key == 'p')
            key = cv::waitKey(0);

        if (key == 27)
            break;
    }


    // find max width, height of object (mean accross all the frames)

    int rows = mInput.rows;
    int cols = mInput.cols;
    int obj_width = 0;
    int obj_height = 0;
    for (int i = 0; i < instance.size(); i++)
    {
        struct contours_info info = instance[i];
        int max_index = 0;
        double max_area = 0;
        for (int j = 0; j < info.boxes.size(); j++)
        {
            double area = info.boxes[j].width * info.boxes[j].height;
            if (area > max_area)
            {
                max_index = j;
                max_area = area;
            }
        }
        cv::Rect max_dim = info.boxes[max_index];

        obj_width = std::max(obj_width, max_dim.width);
        obj_height = std::max(obj_height, max_dim.height);
    }

    cv::Size S = cv::Size(obj_width, obj_height);

    cv::VideoWriter outputVideo; // Open the output
    outputVideo.open(outvid, ex, inputVideo.get(CV_CAP_PROP_FPS), S, true);
    if (!outputVideo.isOpened())
    {
        printf("Could not open the output video for write: %s\n", mFilename.c_str());
        return -1;
    }
    // crop frames and save to video
    for (int i = 0; i < instance.size(); i++)
    {
        struct contours_info info = instance[i];
        cv::Mat outFrame = outFrames[i];
        int rows = outFrame.rows;
        int cols = outFrame.cols;
        cv::Point icentroid = cv::Point(cols / 2, rows / 2);
        cv::Point bcentroid = cv::Point(obj_width / 2, obj_height / 2);

        cv::Rect crop_box;
        crop_box.x = icentroid.x - bcentroid.x;
        crop_box.y = icentroid.y - bcentroid.y;
        crop_box.width = obj_width;
        crop_box.height = obj_height;

        // printf("crop_box: {%d, %d, %d, %d} with rows:%d, cols:%d\n", crop_box.x, crop_box.y, crop_box.width, crop_box.height, rows, cols);
        cv::Mat cropped = outFrame(crop_box).clone();
        
        
        outputVideo << cropped;
        cv::imshow("cropped", cropped);
        char key = cv::waitKey(20);
        if (key == 'p')
            key = cv::waitKey(0);

        if (key == 27)
            break;
    }

    return 0;
}


// --------- Display stuffs ----------
void Scan::draw()
{
    vis_in = mInput.clone();

    int rows = mInput.rows;
    int cols = mInput.cols;

    cv::RNG rng(12345);

    for (int i = 0; i < contours_info.pts.size(); i++)
    {
        cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(150,255));
        cv::drawContours(vis_in, contours_info.pts, i, color, 2, 8, std::vector<cv::Vec4i>(), 0, cv::Point());

        // cv::Rect bbox = contours_boxes[i];
        // if ((bbox.x + bbox.width >= cols) ||
        //     (bbox.y + bbox.height >= rows) ||
        //     (bbox.x <= 0) ||
        //     (bbox.y <= 0))
        //     cv::rectangle(vis_in, contours_boxes[i].tl(), contours_boxes[i].br(), cv::Scalar(255, 0, 0), 2, 8, 0);
        // else
        //     cv::rectangle(vis_in, contours_boxes[i].tl(), contours_boxes[i].br(), color, 2, 8, 0);

        cv::circle(vis_in, contours_info.centroids[i], 4, color, -1);
    }

    cv::circle(vis_in, cv::Point(cols/2, rows/2), 2, cv::Scalar(0,0,255),-1);
}
void Scan::display()
{
    cv::imshow("Input"  , mInput);
    cv::imshow("Thresh" , mThresh);
    cv::imshow("Gray"   , mGray);
    cv::imshow("Edges"  , mEdges);
    cv::imshow("vis_in" , vis_in);
}