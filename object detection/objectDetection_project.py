import cv2
import numpy as np
#import time
from tracker1 import *
#import serial
import os
os.chdir(r"C:\computer vision\projects")


#ser = serial.Serial('COM12', 9600)
#time.sleep(2) 

font = cv2.FONT_HERSHEY_PLAIN
net = cv2.dnn.readNet(r"recurements\yolov3\yolov3-tiny.weights", r"recurements\yolov3\cfg\yolov3-tiny.cfg")
classes = []
with open(r"recurements\yolov3\coco.names", "r") as f:
      classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def object_in_frame(frame,ch=False):
   
    points=[]
    labels=[]

    height,width,c = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.2)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            
     
            labels.append( label)
            
            confidence = confidences[i]
            color = colors[class_ids[i]]

            cv2.rectangle(frame, (x, y), (x + 2*w, y + 2*h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y - 5), font, 1, color, 2)

            if label=='car':
               points.append([x,y,w,h])
               cv2.rectangle(frame, (x, y), (x + 2*w, y + 2*h), (0,255,0), 2)
               cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y - 5), font, 1, (0,255,0), 2)
            
    if ch==False:
     if labels:
         return labels[0]
    else:
          return points
def cars_in_roi(frame,points):
 #       area = [(180,30),(450,30),(580,440),(60,440)]
        area = [(245,305),(400,305),(530,400),(120,400)]
        cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 255), 2)
        cars_inside_roi = set()
        carIdToDelet = []
        tracker = Tracker()
        boxes_id = tracker.update(points) 
        
        for box_id in boxes_id:
            w, h, x, y,idd = box_id
            
            cv2.circle(frame, (w, h), 4, (0, 255, 0), -1)
            results = cv2.pointPolygonTest(np.array(area, np.int32), (w, h), False)
        
            if results > 0:
                cars_inside_roi.add(idd)
            else:
                carIdToDelet.append(idd)
        
        cars_inside_roi = cars_inside_roi.difference(carIdToDelet)
        carIdToDelet.clear()                          
        a = len(cars_inside_roi)
        cv2.putText(frame, 'nearbody cars in the same lane: ' + str(a), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        
        return a    


cap=cv2.VideoCapture(r"self driving car project/project_video.mp4",0)

while True:
    ret, frame = cap.read()
    if ret==True:
     frame=cv2.resize(frame,(640,  480))
     
     points=object_in_frame(frame,1)
     cars=cars_in_roi(frame,points)
     lable=object_in_frame(frame,0)
     
     #if lable=="car" or lable=="stop sign":
        # send=ser.write('s'.encode())
       #  print("sending:",send)
     cv2.imshow("Image", frame)
     key = cv2.waitKey(1)
     if key ==ord("q"):
        break
    else:
       break
cap.release()
cv2.destroyAllWindows()     