import cv2
from cv2 import cuda
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=False)
parser.add_argument('--number_camera', help="number camera in system. Default = 0", default=0)
parser.add_argument('--play_video', help="Tue/False", default=False)
parser.add_argument('--image', help="Tue/False", default=False)
parser.add_argument('--video_path', help="Path of video file", default="videos/car_on_road.mp4")
parser.add_argument('--image_path', help="Path of image to detect objects", default="images/people.jpg")
parser.add_argument('--verbose', help="To print statements", default=True)
parser.add_argument('--tiny_model', help="Light or full model load", default=0)
parser.add_argument('--speed_video', help="Set speed video", default=1)
parser.add_argument('--size_w', help="Size window width", default=320)
parser.add_argument('--size_h', help="Size window height", default=320)
args = parser.parse_args()
number_cam = 0
statisticArray = []
resultArray = []
resultArrayValue = []
resultArrayCountItem = []


# Load yolo
def load_yolo():
    net = cv2.dnn.readNet("models/full/yolov3.weights", "models/full/yolov3.cfg")
    if int(args.tiny_model) == 1:
        net = cv2.dnn.readNet("models/tiny/yolov3-tiny.weights", "models/tiny/yolov3-tiny.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


def load_image(img_path):
    # image loading
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels


def start_webcam(id_camera=0):
    cap = cv2.VideoCapture(int(args.number_camera))

    return cap


def display_blob(blob):
    '''
        Three images each for RED, GREEN, BLUE channel
    '''
    for b in blob:
        for n, imgb in enumerate(b):
            cv2.imshow(str(n), imgb)


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]]) + " : " + str(round(confs[0] * 100, 2))
            statisticArray.append([str(classes[class_ids[i]]), float(round(confs[0] * 100, 2))])
            color = (0, 255, 0)
            if i >= 80:
                color = colors[79]
            # else:
            # color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    cv2.imshow("Out. Press Esc to exit", img)


def image_detect(img_path):
    model, classes, colors, output_layers = load_yolo()
    image, height, width, channels = load_image(img_path)
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    draw_labels(boxes, confs, colors, class_ids, classes, image)
    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break


def webcam_detect(id_cam=0):
    model, classes, colors, output_layers = load_yolo()
    cap = start_webcam()
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()

frame_number = [0]
stop = [False]
def start_video(video_path):
    model, classes, colors, output_layers = load_yolo()
    cap = cv2.VideoCapture(video_path)
    while True:
        if stop[0] == False:
            _, frame = cap.read()
            if frame_number[0] == int(args.speed_video):
                frame_number[0] = 0

                height, width, channels = frame.shape
                blob, outputs = detect_objects(frame, model, output_layers)
                boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
                draw_labels(boxes, confs, colors, class_ids, classes, frame)
            else:
                frame_number[0] = frame_number[0] + 1

        key = cv2.waitKey(1)
        if key == 32:
            if stop[0] == True:
                stop[0] = False
            else:
                stop[0] = True

        if key == 27:
            break
    cap.release()


def init_array():
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    for item in classes:
        resultArray.append(item)
        resultArrayValue.append(0)
        resultArrayCountItem.append(0)


def create_statistic():
    for item in statisticArray:
        index_item = resultArray.index(item[0])
        value = resultArrayValue[index_item] + item[1]
        resultArrayValue[index_item] = value
        resultArrayCountItem[index_item] = resultArrayCountItem[index_item] + 1

    for item in resultArray:
        index = resultArray.index(item)
        if resultArrayCountItem[index] != 0:
            print(
                f'DETECT OBJECT: {item} WITH mean accuracy: {str(round(resultArrayValue[index] / resultArrayCountItem[index], 2))}'
            )


if __name__ == '__main__':
    # cuda.printCudaDeviceInfo(1)
    init_array()
    webcam = args.webcam
    video_play = args.play_video
    image = args.image
    if webcam:
        if args.verbose:
            print('---- Starting Web Cam object detection ----')
        number_cam = args.number_camera
        print(f'---- # camera:{number_cam} ----')
        webcam_detect(number_cam)
    if video_play:
        video_path = args.video_path
        if args.verbose:
            print('Opening ' + video_path + " .... ")
        start_video(video_path)
    if image:
        image_path = args.image_path
        if args.verbose:
            print("Opening " + image_path + " .... ")
        image_detect(image_path)

    create_statistic()
    cv2.destroyAllWindows()
