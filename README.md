# pyolo
## About
python wrapper for darknet yolo detector

## Requirements
opencv
cuda (edit Makefile if not using)
cudnn (edit Makefile if not using)
learnt weight file to run the demo. see demo.py.

## Quick start
1. build the python module as follow.
     ```
     $ git clone --recursive https://github.com/yukikawana/pyolo.git
     $ cd pyolo
     $ make
     ```
2. import the module, initialize the net and pass the image to the net.
     ```python
     import pyolo
     import cv2
     
     pyolo.init('coco_pyolo.data', './darknet/cfg/yolo.cfg', 'yolo.weights')
     img = cv2.imread('input_image.png')
     result = pyolo.predict(img)
     left = result[2]
     top = result[4]
     right = result[3]
     bottom = result[5]
     cv2.rectangle(frame0, (left, top), (right, bottom), (0, 0, 255), 2)
     cv2.imshow('result', img)
     ```

