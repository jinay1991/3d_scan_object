#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_jinay_glosampleapp_MainActivity_image_processing_main(
        JNIEnv *env,
        jobject /* this */,
        jlong addrInRGBA,
        jlong addrOutRGBA) {


    cv::Mat& inRGBA = *(cv::Mat*) addrInRGBA;
    cv::Mat& outRGBA = *(cv::Mat*) addrOutRGBA;

    cv::Mat gray, edges;
    cv::cvtColor(inRGBA, gray, CV_RGBA2GRAY);
    cv::Canny(gray, edges, 80, 170);

    cv::cvtColor(edges, outRGBA, CV_GRAY2RGBA);

    return 1;
}
