import os
import glob
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hvplot.pandas
from skimage import io

#NOT WORK! FOR WORK REQUIRED CUDA DRIVER,VS C++ 14 AND DATASET.

data_dir = Path('F:/kaggle_dataset')
im_list = sorted(data_dir.glob('train_00_part/*.jpg'))

def test_1():
    global im_list, i, img
    im_list = sorted(data_dir.glob('train_00_part/*.jpg'))
    mask_list = sorted(data_dir.glob('train-masks-f/*.png'))
    boxes_df = pd.read_csv(data_dir / 'oidv6-train-annotations-bbox.csv')
    names_ = ['LabelName', 'Label']
    labels = pd.read_csv(data_dir / 'class-descriptions-boxable.csv', names=names_)
    im_ids = [im.stem for im in im_list]
    cols = ['ImageID', 'LabelName', 'XMin', 'YMin', 'XMax', 'YMax']
    boxes_df = boxes_df.loc[boxes_df.ImageID.isin(im_ids), cols] \
        .merge(labels, how='left', on='LabelName')
    print(boxes_df)
    # Annotate and plot
    cols, rows = 3, 2
    plt.figure(figsize=(20, 30))
    for i, im_file in enumerate(im_list[9:15], start=1):
        df = boxes_df.query('ImageID == @im_file.stem').copy()
        img = cv2.imread(str(im_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Add boxes
        h0, w0 = img.shape[:2]
        coords = ['XMin', 'YMin', 'XMax', 'YMax']
        df[coords] = (df[coords].to_numpy() * np.tile([w0, h0], 2)).astype(int)

        for tup in df.itertuples():
            cv2.rectangle(img, (tup.XMin, tup.YMin), (tup.XMax, tup.YMax),
                          color=(0, 255, 0), thickness=2)
            cv2.putText(img, tup.Label, (tup.XMin + 2, tup.YMax - 2),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=1, color=(0, 255, 0), thickness=2)

        # Add segmentation masks
        mask_files = [m for m in mask_list if im_file.stem in m.stem]
        mask_master = np.zeros_like(img)
        np.random.seed(10)
        for m in mask_files:
            mask = cv2.imread(str(m))
            mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_AREA)
            color = np.random.choice([0, 255], size=3)
            mask[np.where((mask == [255, 255, 255]).all(axis=2))] = color
            mask_master = cv2.add(mask_master, mask)
        img = cv2.addWeighted(img, 1, mask_master, 0.5, 0)

        plt.subplot(cols, rows, i)
        plt.axis('off')
        plt.imshow(img)
    # plt.show()


# test_1()

urls = pd.read_csv(data_dir / "image_ids_and_rotation.csv",
                   usecols=['ImageID', 'OriginalURL'])
print(urls)

classes = np.loadtxt(data_dir / "openimages.names", dtype=np.str, delimiter="\n")
net = cv2.dnn.readNet(str(data_dir / "yolov3-openimages.weights"), str(data_dir / "yolov3-openimages.cfg"))

layer_names = net.getLayerNames()
outputlayers = []
for i in net.getUnconnectedOutLayers():
    ind = i
    outputlayers.append(layer_names[ind-1])

#outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

im_url = urls.loc[urls.ImageID == im_list[11].stem, 'OriginalURL'].squeeze()
img = io.imread(im_url)

height, width, channels = img.shape

# Make a blob array and run it through the network
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(outputlayers)

# Get confidence scores and objects
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.2:  # threshold
            print(confidence)
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])  # put all rectangle areas
            confidences.append(float(confidence))  # how confidence was that object detected and show that percentage
            class_ids.append(class_id)  # name of the object tha was detected

# Non-max suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)
print(indexes, boxes, class_ids)
