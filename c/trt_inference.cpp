//
// Created by shiby on 23-9-24.
//

#include "trt_inference.h"


class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;

std::vector<std::string> loadClasses(std::string &classPath) {
    std::vector<std::string> classes;
    std::string name;
    std::ifstream fp(classPath);
    if (!fp.is_open()) {
        std::cout << "Cannot find file: " << classPath << std::endl;
        exit(-1);
    }
    while (!fp.eof()) {
        std::getline(fp, name);
        if (name.length() > 0)
            classes.push_back(name);
    }
    fp.close();

    return classes;
}

void preprocess_image(cv::Mat &frame, int input_w, int input_h, InputData &input) {
    cv::Mat src_image;
    cv::cvtColor(frame, src_image, cv::COLOR_BGR2RGB);
    int w = src_image.cols;
    int h = src_image.rows;
    int max_size = std::max(h, w);
    cv::Mat image = cv::Mat::zeros(cv::Size(max_size, max_size), CV_8UC3);
    cv::Rect roi(0, 0, w, h);
    src_image.copyTo(image(roi));

    input.x_factor = image.cols / static_cast<float>(input_w);
    input.y_factor = image.rows / static_cast<float>(input_h);
    input.image = cv::dnn::blobFromImage(image, 1.0 / 255., cv::Size(input_w, input_h));
}

void postprocess(cv::Mat &output, float conf_threshold, float score_threshold, float x_factor, float y_factor,
                 std::vector<DetectedResult> &result) {

    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;

    int output_w = output.cols;
    for (int i = 0; i < output.rows; ++i) {
        float conf = output.at<float>(i, 4);
        if (conf < conf_threshold)
            continue;
        cv::Mat class_scores = output.row(i).colRange(5, output_w);
        cv::Point classIdPoint;
        double score;
        cv::minMaxLoc(class_scores, 0, &score, 0, &classIdPoint);
        if (score > score_threshold) {
            float cx = output.at<float>(i, 0);
            float cy = output.at<float>(i, 1);
            float w = output.at<float>(i, 2);
            float h = output.at<float>(i, 3);
            int x = static_cast<int>((cx - 0.5 * w) * x_factor);
            int y = static_cast<int>((cy - 0.5 * h) * y_factor);
            int width = static_cast<int>(w * x_factor);
            int height = static_cast<int>(h * y_factor);
            cv::Rect box(x, y, width, height);
            boxes.push_back(box);
            classIds.push_back(classIdPoint.x);
            confidences.push_back(static_cast<float>(score));
        }
    }

    // NMS
    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
    for (auto idx: indexes) {
        DetectedResult dr{classIds[idx], confidences[idx], boxes[idx]};
        result.emplace_back(dr);
    }
}


int main(int argc, char **argv) {

    std::string labelPath = "../example/uav_bird.txt";
    std::vector<std::string> labels = loadClasses(labelPath);
    std::string enginPath = "../uav_bird.engine";
    std::ifstream fEngine(enginPath, std::ios::binary);
    char *trtModelStream = nullptr;

    int size = 0;
    if (fEngine.good()) {
        fEngine.seekg(0, std::ifstream::end);
        size = fEngine.tellg();
        fEngine.seekg(0, std::ifstream::beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        fEngine.read(trtModelStream, size);
        fEngine.close();
    }


    // Create runtime
    auto runtime = nvinfer1::createInferRuntime(gLogger);
    assert(runtime != nullptr);
    auto engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    auto context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    void *buffers[2] = {nullptr, nullptr};
    std::vector<float> output;
    cudaStream_t stream;


    // NCHW
    nvinfer1::Dims input_shape = engine->getTensorShape("images");
    nvinfer1::Dims output_shape = engine->getTensorShape("output0");


    int input_h = input_shape.d[2];
    int input_w = input_shape.d[3];

    int output_h = output_shape.d[1];
    int output_w = output_shape.d[2];

    cudaMalloc(&buffers[0], input_h * input_w * 3 * sizeof(float));
    cudaMalloc(&buffers[1], output_h * output_w * sizeof(float));

    output.resize(output_h * output_w);

    cudaStreamCreate(&stream);

    cv::Mat image = cv::imread("../example/example.jpg");
    InputData input;
    preprocess_image(image, input_h, input_w, input);

    // async
    cudaMemcpyAsync(buffers[0], input.image.ptr<float>(), input_h * input_w * 3 * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // inference
    context->enqueueV2(buffers, stream, nullptr);

    // async
    cudaMemcpyAsync(output.data(), buffers[1], output_h * output_w * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

    cv::Mat output_mat(output_h, output_w, CV_32F, (float *) output.data());
    std::vector<DetectedResult> result;
    postprocess(output_mat, 0.25f, 0.25f, input.x_factor, input.y_factor, result);

    for (const auto res: result) {
        cv::putText(image, labels[res.class_id], cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX,
                    1.0, cv::Scalar(255, 0, 0), 2, 8);
        cv::rectangle(image, res.box, cv::Scalar(255, 0, 0));
    }

    cv::imshow("Image", image);
    cv::waitKey(0);

    // deallocate
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);


    delete context;
    delete engine;
    delete runtime;

    return 0;
}