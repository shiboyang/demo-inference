cmake_minimum_required(VERSION 3.10)

project(inference-demo VERSION 1.0)


find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)


include_directories(
        ${OpenCV_INCLUDE_DIRS}
        /usr/local/Tensorrt/include
        ${CUDA_INCLUDE_DIRS}
)

link_directories(
        /usr/local/Tensorrt/lib
)

add_executable(trt_inference
        c/trt_inference.cpp
        )

add_executable(infer2
        main.cpp
        c/trt_inference2.cpp
        )


target_link_libraries(trt_inference PRIVATE
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        nvinfer
        nvinfer_plugin
        )


target_link_libraries(infer2 PRIVATE
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        nvinfer
        nvinfer_plugin
        )
