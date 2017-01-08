#!/usr/bin/python
import pyolo
import cv2 as cv
# import numpy as np


cap0 = cv.VideoCapture(0)
ini = 'coco_pyolo.data'
cfg = './darknet/cfg/yolo.cfg'
w = '../darknet/yolo.weights'
pyolo.init(ini, cfg, w)
while(cap0.isOpened()):
    ret0, frame0 = cap0.read()
    cv.imwrite("frame0.png",frame0)
    a = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
    ret = pyolo.predict(a)
    if len(ret) < 6:
        continue
    if min(ret) < 0:
        print min(ret)
        continue
    cv.rectangle(frame0, (ret[2], ret[4]), (ret[3], ret[5]), (0, 0, 255), 2)
    cv.imshow('frame0', frame0)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap0.release()
cv.destroyAllWindows()
