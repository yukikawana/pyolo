#!/usr/bin/python
import pyolo
import cv2 as cv
# import time
# import numpy as np


ini = 'coco_pyolo.data'
cfg = './darknet/cfg/yolo.cfg'
w = '../darknet/yolo.weights'
pyolo.init(ini, cfg, w)
while(True):
    frame0 = cv.imread("frame0.png")
    ret = pyolo.predict(frame0)
    if len(ret) < 6:
        continue
    for re in range(len(ret) / 6):
        rele = ret[re * 6:re * 6 + 6]
        print 'len in python ', len(rele)
        if min(rele) < 0:
            print 'min ', min(rele)
            continue
        left = rele[2]
        top = rele[4]
        right = rele[3]
        bottom = rele[5]
        cv.rectangle(frame0, (left, top), (right, bottom), (0, 0, 255), 2)
    # cv.imshow('frame0', frame0)
    # if cv.waitKey(1) & 0xFF == ord('q'):
    #    break
