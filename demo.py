#!/usr/bin/python
import pyolo
import cv2 as cv

cap0 = cv.VideoCapture(0)

ini = 'coco_pyolo.data'  # object category info, etc.
cfg = './darknet/cfg/yolo.cfg'  # network structure
w = './darknet/yolo.weights'  # lerant weight. need to wget http://pjreddie.com/media/files/yolo.weights
pyolo.init(ini, cfg, w)  # initialize the net

while(cap0.isOpened()):
    ret0, frame0 = cap0.read()
    result = pyolo.predict(frame0)

    if not len(result) > 0:  # if not object is detected continue
        continue

    for id in range(len(result)):
        result_by_id = result[id]
        if min(result_by_id) < 0:
            continue

        left = result_by_id[2]
        top = result_by_id[4]
        right = result_by_id[3]
        bottom = result_by_id[5]
        cv.rectangle(frame0, (left, top), (right, bottom), (0, 0, 255), 2)
        cv.imshow('frame0', frame0)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap0.release()
cv.destroyAllWindows()
