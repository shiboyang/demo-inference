//
// Created by shiby on 23-9-28.
//

#ifndef INFERENCE_DEMO_TRT_INFERENCE2_H
#define INFERENCE_DEMO_TRT_INFERENCE2_H

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <memory>
#include <iostream>

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


class TRTInference {

public:

    void init(std::string engine_path,
              float confidence_threshold,
              float score_threshold,
              float nms_threshold,
              float nms_score_threshold);

    void infer(cv::Mat &frame, std::vector<DetectedResult> &result);

    void preprocess(cv::Mat &frame, InputData &input);

    void postprocess(cv::Mat &output, float x_factor, float y_factor, std::vector<DetectedResult> &result);

private:
    nvinfer1::IRuntime *mRuntime;
    nvinfer1::ICudaEngine *mEngine;
    nvinfer1::IExecutionContext *mContext;
    float mConfThreshold;
    float mScoreThreshold;
    float mNMSScoreThreshold;
    float mNMSThreshold;


    int mInputH;
    int mInputW;
    int mOutputH;
    int mOutputW;

    cudaStream_t mStream;

    void *mBuffers[2]{nullptr, nullptr};
};


#endif //INFERENCE_DEMO_TRT_INFERENCE2_H
