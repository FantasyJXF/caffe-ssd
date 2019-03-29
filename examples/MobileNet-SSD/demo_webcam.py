# -*- coding=utf-8 -*-

import numpy as np
import sys,os
import cv2
import time
caffe_root = '/home/jxf/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

net_file= 'deploy.prototxt'  
caffe_model='mobilenet_iter_73000.caffemodel'  
test_dir = "images"

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()

caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(net_file,caffe_model,caffe.TEST)

CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])
    
    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(frame):
    ##  origimg = cv2.imread(imgfile)
    img = preprocess(frame)
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    ##统计时间
    net.blobs['data'].data[...] = img
    start=time.time()
    out = net.forward()
    use_time=time.time()-start
    print ("time="+str(use_time)+"s") 
    fps=1/use_time
    print ("FPS="+str(fps)) 
    box, conf, cls = postprocess(frame, out)
    ##调摄像头
    ##  k = cv2.waitKey(30) & 0xff
    ##  cap.release()
    ##cv2.destroyAllWindows() 
    #Exit if ESC pressed
    ##  if k == 27 : return False
    return True

cap = cv2.VideoCapture(0)
while(1):
    ret, frame = cap.read()
    ##detect(frame)
    img = preprocess(frame)
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    net.blobs['data'].data[...] = img
    start=time.time()
    out = net.forward()
    use_time=time.time()-start
    #print ("time="+str(use_time)+"s")
    fps=1/use_time
    #print ("FPS="+str(fps))
    box, conf, cls = postprocess(frame, out)

    for i in range(len(box)):
        p1 = (box[i][0], box[i][1])
        p2 = (box[i][2], box[i][3])
        cv2.rectangle(frame, p1, p2, (0,255,0))
        p3 = (max(p1[0], 15), max(p1[1], 15))
        title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
        cv2.putText(frame, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    cv2.imshow("SSD", frame)
    #cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 
