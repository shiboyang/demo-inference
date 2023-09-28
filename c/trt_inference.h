//
// Created by shiby on 23-9-27.
//

#ifndef INFERENCE_DEMO_TRT_INFERENCE_H
#define INFERENCE_DEMO_TRT_INFERENCE_H

#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

struct DetectedResult {
    int class_id;
    double confidence;
    cv::Rect box;
};

struct InputData {
    float x_factor = 1.0f;
    float y_factor = 1.0f;
    cv::Mat image;
};

#endif //INFERENCE_DEMO_TRT_INFERENCE_H
