#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_cuda.h>

#include <iostream>
#include <cstdio>
#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace cvf {
    cv::Mat cvReadImg(const char* filename, unsigned char*& buffer);
    void cvOutputImg(const char* filename, cv::Mat& image, unsigned char*& buffer);
};

namespace CUDAf {
    void getDeviceName();
    __global__ void dMedianFilter(const unsigned int inputH, const unsigned int inputW, const unsigned int outputH, const unsigned int outputW, const unsigned int kSize, const unsigned int pad, const unsigned int cout, unsigned char* X, unsigned char* Y);
    __global__ void dMeanFilter(const unsigned int inputH, const unsigned int inputW, const unsigned int outputH, const unsigned int outputW, const unsigned int kSize, const unsigned int pad, const unsigned int cout, unsigned char* X, unsigned char* Y);
};
