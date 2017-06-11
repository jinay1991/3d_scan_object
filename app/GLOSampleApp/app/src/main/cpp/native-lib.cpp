#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <scan.hpp>

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_jinay_glosampleapp_MainActivity_image_1processing_1main(JNIEnv *env,
                                                                         jobject instance,
                                                                         jlong addrInRGBA,
                                                                         jlong addrOutRGBA) {

    cv::Mat inRGBA = *(cv::Mat *) addrInRGBA;
    cv::Mat outRGBA = *(cv::Mat *) addrOutRGBA;

    cv::Mat bgr;
    cv::cvtColor(inRGBA, bgr, CV_RGBA2BGR);

    Scan scn(bgr);
    scn.preprocess();
    scn.detect_edges();
    scn.detect_contours();
    scn.draw();

    cv::cvtColor(scn.vis_in, outRGBA, CV_BGR2RGBA);

    return (jint) 1;

}