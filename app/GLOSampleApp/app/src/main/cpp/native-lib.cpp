#include "native-lib.h"

JNIEXPORT jint JNICALL
Java_com_example_jinay_glosampleapp_MainActivity_scanner(JNIEnv *env,
                                                                         jobject instance,
                                                                         jlong addrInRGBA,
                                                                         jlong addrOutRGBA) {

    cv::Mat inRGBA = *(cv::Mat *) addrInRGBA;
    cv::Mat outRGBA = *(cv::Mat *) addrOutRGBA;

    ProcessMain(inRGBA, outRGBA);

    // Scan scn(bgr);
    // scn.Image();

    // cv::cvtColor(scn.vis_in, outRGBA, CV_BGR2RGBA);


    return (jint) 1;

}

int ProcessMain(cv::Mat& in, cv::Mat& out)
{
    cv::Mat bgr;
    cv::cvtColor(in, bgr, CV_GRAY2BGR);

    Scan scn(bgr);
    scn.Image(false);

    cv::cvtColor(scn.vis_in, out, CV_BGR2RGB);

    return 0;
}