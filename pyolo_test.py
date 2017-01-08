#!/usr/bin/python
import pyolo as rc
import cv2 as cv
import os.path
import os
a = cv.imread('../darknet/data/download.jpg')
# a = cv.imread('data/alto.jpg')
a = cv.cvtColor(a, cv.COLOR_BGR2RGB)
print a.dtype
ini = 'coco_pyolo.data'
cfg = './darknet/cfg/yolo.cfg'
# if not os.path.isfile('yolo.weights'):
#    os.system('wget http://pjreddie.com/media/files/yolo.weights')
w = '../darknet/yolo.weights'
rc.init(ini, cfg,w)

ret = rc.predict(a)
print ''
print 'predicted ', len(ret)
print ret
b = cv.imread('../darknet/data/download.jpg')
a = cv.cvtColor(b, cv.COLOR_BGR2RGB)
ret = rc.predict(a)
print ''
print 'predicted ', len(ret)
print ret
print ret[0]
print ret[1]
print ret[2]
print ret[3]
print ret[4]
print ret[5]
cv.rectangle(b, (ret[2], ret[4]), (ret[3], ret[5]), (0, 0, 255), 2)
cv.imwrite("a.png", b)
