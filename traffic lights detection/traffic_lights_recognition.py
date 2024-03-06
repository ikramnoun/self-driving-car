import cv2
import numpy as np
import os

os.chdir(r"C:\computer vision\projects")

HEIGHT = 32
WIDTH = 32

net = cv2.dnn.readNet(r"recurements\yolov3\yolov3-tiny.weights", r"recurements\yolov3\cfg\yolov3-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

classes = []
with open("recurements\yolov3\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()] 


font = cv2.FONT_HERSHEY_PLAIN



def classification(croped_img,original_img,x,y,w,h):
  
    hsv = cv2.cvtColor(croped_img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([150, 70, 50])
    upper_red = np.array([180, 255, 255])

    lower_yellow = np.array([21, 39, 64])
    upper_yellow = np.array([40, 255, 255])
    
    lower_green = np.array([60, 100, 100])
    upper_green = np.array([80, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    if mask_red.any():
        cv2.rectangle(original_img, (x-10, y-15), (x + w+10, y + h+20), (0,0,255), 2)
        cv2.putText(original_img,"stop", (x, y-20), font, 0.75, (0, 0, 255), 2)

    if mask_yellow.any():
        cv2.rectangle(original_img, (x-10, y-15), (x + w+10, y + h+20), (0,255,255), 2)
        cv2.putText(original_img,"stop", (x, y-20), font, 0.75, (0,255, 255), 2)

    if mask_green.any(): 
        cv2.rectangle(original_img, (x-10, y-15), (x + w+10, y + h+20), (0,255,0), 2)
        cv2.putText(original_img,"go", (x, y-20), font, 0.75, (0, 255,0), 2)

def traffic_lights(frame,ch):
     
     lights=[]

     height, width, channels = frame.shape

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

     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

     for i in range(len(boxes)):
        if i in indexes:
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            x,y,w,h = boxes[i]
        
            if label=="traffic light":
                '''crop the detected signs -> input to the classification model'''
                crop = frame[y-10:y+h+20, x-10:x+w+10]
                crop=cv2.resize(crop,(32,32))
                if len(crop) >0:
                        light=classification(crop,frame,x,y,w,h)
                        lights.append(light)
     if ch==0:
         if lights:
             return lights[0]

cap=cv2.VideoCapture(r"C:\computer vision\projects\recurements\Grays - GEC Elliot, Plessey Mellors & PEEK Elite - Microsense Pelican Crossing Traffic Lights, Essex.mp4")

while True:
    ret, frame=cap.read()
    frame=cv2.resize(frame,(720,640))
    if ret==True:
     light=traffic_lights(frame,0)
     cv2.imshow( "frame",frame)
     key = cv2.waitKey(1)
     if key ==ord("q"):
          break
    else:
        break
cap.release()
cv2.destroyAllWindows()