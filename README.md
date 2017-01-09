# pyolo
## About
python wrapper for darknet yolo detector

## Requirements
* python 2.x
* opencv (to run demo.py)
* cuda (edit Makefile if not using)
* cudnn (edit Makefile if not using)
* boost numpy
* boost python
* learnt weight file to run the demo.

## Quick start
1. build the python module as follow.  
     ```
     $ git clone --recursive https://github.com/yukikawana/pyolo.git  
     $ cd pyolo  
     $ make  
    ```
    
2. get the learnt weight of yolo. 
     ```
     $ cd darknet
     $ wget http://pjreddie.com/media/files/yolo.weights
     ```
     
3. get some random input image, import the module, initialize the net and pass the image to the net.
     ```python
     import pyolo
     import cv2
     
     pyolo.init('coco_pyolo.data', './darknet/cfg/yolo.cfg', 'yolo.weights')
     img = cv2.imread('./darknet/data/person.jpg')
     pyolo.predict(img)
     result_by_id = result[0]
     left = result_by_id[2]
     top = result_by_id[4]
     right = result_by_id[3]
     bottom = result_by_id[5]
     cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
     cv2.imshow('detected object', img)
     ```
     see more example in demo.py

