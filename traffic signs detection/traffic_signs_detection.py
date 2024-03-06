import cv2
import numpy as np
from keras.models import load_model
import os

os.chdir(r"C:\computer vision\projects")

HEIGHT = 32
WIDTH = 32

net = cv2.dnn.readNet(r"recurements\yolov4-tiny_training_last.weights", r"recurements\yolov4-tiny_training.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

classes = []

with open("recurements/signs.names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
 
classification_model = load_model(r"recurements/traffic_signs.h5") 
classes_classification = []
with open("recurements/signs_classes.txt", "r") as f:
    classes_classification = [line.strip() for line in f.readlines()]


color =(255,0, 255)
font = cv2.FONT_HERSHEY_PLAIN


def preprocessing(img):
 img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 img = cv2.equalizeHist(img)
 img = img/255
 return img

def classification(img,model,classes):
   img = np.asarray(img)
   img = cv2.resize(img, (32, 32))
   img = preprocessing(img)
   img = img.reshape(1, 32, 32, 1)
   
   predictions = model.predict(img)
   classIndex = np.argmax(predictions)
   probabilityValue =np.amax(predictions)
   if probabilityValue > 0.7:
      label= str(classes[classIndex])
      return label

def traffic_signs(frame,ch):
     
     sings=[]

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
            if confidence > 0.5:
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

     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

     for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            '''crop the detected signs -> input to the classification model'''
            crop_img = frame[y:y+h, x:x+w]
            if len(crop_img) >0:
                    sign=classification(crop_img,classification_model,classes_classification)
                    sings.append(sign)
                    cv2.putText(frame,sign, (x, y-5), font, 0.75, (255, 0, 255), 2)
     if ch==0:
         if sings:
             return sings[0]

cap=cv2.VideoCapture(r"recurements/traffic-sign-to-test.mp4",0)

while True:
    ret, frame=cap.read()
    frame=cv2.resize(frame,(800,600))
    if ret==True:
     sign=traffic_signs(frame,0)
     cv2.imshow("Image", frame)
     key = cv2.waitKey(1)
     if key ==ord("q"):
          break
    else:
        break
cap.release()
cv2.destroyAllWindows()