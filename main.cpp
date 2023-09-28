//
// Created by shiby on 23-9-28.
//
#include "c/trt_inference2.h"
#include "c/utils.h"


int main() {
    std::string engine_path = "../asset/uav_bird_fp16.engine";
    std::string video_path = "../example/bird.mp4";
    std::string label_path = "../example/uav_bird.txt";
    auto detector = std::make_shared<TRTInference>();
    std::vector<std::string> labels;
    std::vector<DetectedResult> results;

    labels = loadClasses(label_path);
    detector->init(engine_path, 0.25, 0.25, 0.25, 0.45);
    cv::VideoCapture cap(video_path);
    cv::Mat frame;
    while (true) {
        bool ret = cap.read(frame);
        if (!ret) break;
        detector->infer(frame, results);

        for (const auto res: results) {
            cv::putText(frame, labels[res.class_id], cv::Point(res.box.tl().x, res.box.tl().y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0));
            cv::rectangle(frame, res.box, cv::Scalar(255, 0, 0));
        }
        cv::imshow("Image", frame);
        char c = cv::waitKey(60);
        if (c == 27) break;
        results.clear();
    }

    return 0;
}