//
// Created by Jinay Patel on 13/06/17.
//

#ifndef GLOSAMPLEAPP_NATIVE_LIB_H
#define GLOSAMPLEAPP_NATIVE_LIB_H
#include <jni.h>
#include <string>

#include <opencv2/opencv.hpp>
#include <scan.hpp>

int ProcessMain(cv::Mat& in, cv::Mat& out);

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_jinay_glosampleapp_MainActivity_scanner(JNIEnv *env,
                                                                         jobject instance,
                                                                         jlong addrInRGBA,
                                                                         jlong addrOutRGBA);


#endif //GLOSAMPLEAPP_NATIVE_LIB_H
