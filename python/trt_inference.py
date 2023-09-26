import time
import cv2

import numpy as np
import torch
from torchvision import transforms
from collections import OrderedDict, namedtuple
from typing import AnyStr, List, Tuple
import tensorrt as trt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(640),
])


def load_classes(classes_path: AnyStr) -> List[AnyStr]:
    """
    Load class list from txt file.
    :param classes_path:
    :return:
    """
    classes = []
    with open(classes_path, "r") as f:
        classes = [cname.strip() for cname in f.readlines()]

    return classes


def image_preprocess(img: np.ndarray) -> np.ndarray:
    """
    Preprocess image.
    :param img:
    :return:
    """
    rows, cols, _ = img.shape
    size = max(rows, cols)
    result = np.zeros([size, size, 3], np.uint8)
    result[0:rows, 0:cols] = img
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result


def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []
    # print(output_data.shape)
    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / 640.0
    y_factor = image_height / 640.0

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.25:
            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)
            if (classes_scores[class_id] > .25):
                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.25)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes


def trt_inference():
    """
    TensorRT Inference API
    :return:
    """
    classes = load_classes("./uav_bird.txt")
    device = torch.device("cuda:0")
    Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
    bindings = OrderedDict()
    trt_logger = trt.Logger(trt.Logger.INFO)
    with open("../uav_bird.engine", "rb") as f, trt.Runtime(trt_logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())
    for i in range(model.num_bindings):
        name = model.get_tensor_name(i)
        dtype = trt.nptype(model.get_tensor_dtype(name))
        shape = model.get_tensor_shape(name)
        data = torch.from_numpy(np.zeros(shape, dtype=dtype)).to(device)
        bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))

    binding_address = OrderedDict((n, d.ptr) for n, d in bindings.items())
    trt_context = model.create_execution_context()

    capture = cv2.VideoCapture("./bird.mp4")
    colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

    while True:
        flag, frame = capture.read()
        if flag is False:
            break
        start = time.time()
        image = image_preprocess(frame)
        input_img = transform(image).view(1, 3, 640, 640).to(device)
        binding_address["images"] = input_img.data_ptr()
        trt_context.execute_v2(list(binding_address.values()))
        output = bindings["output0"].data.cpu().numpy()
        end = time.time()

        class_ids, confidences, boxes = wrap_detection(image, np.squeeze(output, 0))
        for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            color = colors[int(classid) % len(colors)]
            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(frame, classes[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

        inf_end = end - start
        fps = 1 / inf_end
        fps_label = "FPS: %.2f" % fps
        cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("YOLOv5s 7.0 + TensorRT8.6.x by gloomyfish", frame)
        cc = cv2.waitKey(1)
        if cc == 27:
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    trt_inference()


if __name__ == '__main__':
    main()
