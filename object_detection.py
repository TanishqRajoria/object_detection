import mediapipe as mp
import os
import cv2

vid=cv2.VideoCapture(0)
vid.set(3,640)
vid.set(4,480)


classNames=[]
classFiles='coco.names'

with open(classFiles,'rt') as f:
    classNames=f.read().rstrip('\n').split("\n")

configPath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath='frozen_inference_graph.pb'
net=cv2.dnn.DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)


while True:

    _,img=vid.read()
    classIds,confs,bbox=net.detect(img,confThreshold=0.5)
    print(classIds,bbox)
    if len(classIds)!=0:
        for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
            cv2.putText(img,str(confidence*100),(box[0]+150,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

        cv2.imshow("Camera",img)
        key=cv2.waitKey(100)

        if key==27:
            break
vid.release()
cv2.destroyAllWindows()