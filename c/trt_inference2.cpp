//
// Created by shiby on 23-9-28.
//

#include "trt_inference2.h"


class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;

void TRTInference::init(std::string engine_path, float confidence_threshold, float score_threshold,
                        float nms_threshold,
                        float nms_score_threshold) {
    std::ifstream infile(engine_path, std::ifstream::binary);
    int size = 0;
    assert(infile.is_open() && "Attempt to reading from a file that is not open.");
    infile.seekg(0, std::ifstream::end);
    size = infile.tellg();
    infile.seekg(0, std::ifstream::beg);
    std::shared_ptr<char> modelStream(new char[size]);

    infile.read(modelStream.get(), size);
    infile.close();

    mRuntime = nvinfer1::createInferRuntime(gLogger);
    mEngine = mRuntime->deserializeCudaEngine(modelStream.get(), size);
    mContext = mEngine->createExecutionContext();

    // NCHW
    nvinfer1::Dims mInputShape = mEngine->getTensorShape("images");
    nvinfer1::Dims mOutputShape = mEngine->getTensorShape("output0");

    mInputH = mInputShape.d[2];
    mInputW = mInputShape.d[3];
    mOutputH = mOutputShape.d[1];
    mOutputW = mOutputShape.d[2];

    cudaMalloc(&mBuffers[0], mInputH * mInputW * 3 * sizeof(float));
    cudaMalloc(&mBuffers[1], mOutputH * mOutputW * sizeof(float));

    cudaStreamCreate(&mStream);

    mScoreThreshold = score_threshold;
    mConfThreshold = confidence_threshold;
    mNMSScoreThreshold = nms_score_threshold;
    mNMSThreshold = nms_threshold;
}


void TRTInference::preprocess(cv::Mat &frame, InputData &input) {
    cv::Mat rgb_image;
    cv::cvtColor(frame, rgb_image, cv::COLOR_BGR2RGB);
    int image_w = frame.cols;
    int image_h = frame.rows;
    int max_size = std::max(image_h, image_w);
    cv::Mat dst_image = cv::Mat::zeros(cv::Size(max_size, max_size), CV_8UC3);
    cv::Rect roi(0, 0, image_w, image_h);
    rgb_image.copyTo(dst_image(roi));


    input.x_factor = dst_image.cols / static_cast<float>(mInputW);
    input.y_factor = dst_image.rows / static_cast<float>(mInputH);
    input.image = cv::dnn::blobFromImage(dst_image, 1.0 / 255.0, cv::Size(mInputW, mInputH));
}


void TRTInference::postprocess(cv::Mat &output, float x_factor, float y_factor, std::vector<DetectedResult> &result) {
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;

    for (int i = 0; i < output.rows; ++i) {
        float conf = output.at<float>(i, 4);
        if (conf < mConfThreshold)
            continue;
        cv::Mat class_scores = output.row(i).colRange(5, mOutputW);
        cv::Point max_location;
        double max_score;
        cv::minMaxLoc(class_scores, 0, &max_score, 0, &max_location);
        if (max_score > mScoreThreshold) {
            float cx = output.at<float>(i, 0);
            float cy = output.at<float>(i, 1);
            float ow = output.at<float>(i, 2);
            float oh = output.at<float>(i, 3);
            int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
            int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
            int w = static_cast<int>(ow * x_factor);
            int h = static_cast<int>(oh * y_factor);
            cv::Rect box(x, y, w, h);
            confidences.push_back(conf);
            class_ids.push_back(max_location.x);
            boxes.push_back(box);
        }
    }

    //NMS
    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, mNMSScoreThreshold,
                      mNMSThreshold, indexes);
    //constructing the detected result.
    for (int idx: indexes) {
        DetectedResult dr{class_ids[idx], confidences[idx], boxes[idx]};
        result.push_back(dr);
    }
}

void TRTInference::infer(cv::Mat &frame, std::vector<DetectedResult> &result) {
    InputData input;
    preprocess(frame, input);
    std::vector<float> output(mOutputH * mOutputW);

    cudaMemcpyAsync(mBuffers[0], input.image.ptr<float>(), mInputH * mInputW * 3 * sizeof(float),
                    cudaMemcpyHostToDevice, mStream);
    mContext->enqueueV2(mBuffers, mStream, nullptr);

    cudaMemcpyAsync(output.data(), mBuffers[1], mOutputH * mOutputW * sizeof(float),
                    cudaMemcpyDeviceToHost, mStream);
    cv::Mat output_mat(mOutputH, mOutputW, CV_32F, output.data());

    postprocess(output_mat, input.x_factor, input.y_factor, result);
}

