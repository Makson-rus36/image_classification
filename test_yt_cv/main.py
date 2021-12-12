import random

import cv2

#img = cv2.imread("lena.png")
videoCap = cv2.VideoCapture(1)
videoCap.set(3, 720)
videoCap.set(3, 480)

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# print(classNames)

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)

net.setInputSize(720, 480)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = videoCap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=0.55)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classIds, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            #color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            color = (0,255,0)
            cv2.rectangle(img, box, color=color, thickness=2)
            cv2.putText(img, "Class: "+classNames[classIds - 1], (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color)

            cv2.putText(img, 'Acc: '+str(round(confidence*100,2))+'%', (box[0]+15, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color)
    cv2.imshow("Output", img)
    cv2.waitKey(1)
