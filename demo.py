#!/usr/bin/python
import pyolo
import cv2 as cv
import time


# width and hight of input for yolo cnn should be 416x416
yolo_width = 416
yolo_hight = 416

cap0 = cv.VideoCapture(0)

ini = 'coco_pyolo.data'  # object category info, etc.
cfg = './darknet/cfg/yolo.cfg'  # network structure
w = './darknet/yolo.weights'  # lerant weight. need to wget http://pjreddie.com/media/files/yolo.weights
names = './darknet/data/coco.names'  # object category list

object_category = []
for line in open(names):
    object_category.append(line.strip('\n'))

pyolo.init(ini, cfg, w)  # initialize the net

while(cap0.isOpened()):
    start = time.time()
    ret0, frame0 = cap0.read()
    im = cv.resize(frame0, (yolo_width, yolo_hight))  # adjust input image size for yolo cnn.
    result = pyolo.predict(im)
    """
    result is a 2-dimensional INT array and each row contains following information in order of
    [class id][confidence][left][right][top][bottom]
    """

    if len(result) > 0:  # if object is detected
        for id in range(len(result)):
            result_by_id = result[id]
            if min(result_by_id) < 0:
                continue

            # scale the coordinate back to the original iamge's one
            xscale = float(frame0.shape[1]) / float(yolo_width)
            yscale = float(frame0.shape[0]) / float(yolo_hight)
            left = int(result_by_id[2] * xscale)
            top = int(result_by_id[4] * yscale)
            right = int(result_by_id[3] * xscale)
            bottom = int(result_by_id[5] * yscale)

            cv.rectangle(frame0, (left, top), (right, bottom), (0, 0, 255), 2)
    cv.imshow('frame0', frame0)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    elapsed = time.time() - start
    print object_category[result_by_id[0]], ': ', result_by_id[1], '%'
    print ("processed at {0}".format(1 / elapsed) + "[fps]")

cap0.release()
cv.destroyAllWindows()
