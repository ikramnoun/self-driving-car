import cv2
import numpy as np
from main_lane_detection import *
from objectDetection_project import *
from Drowsy_Face_detection import *
from traffic_signs_detection import *
from traffic_lights_recognition import *
#import playsound
#from playsound import playsound as ply
import os

os.chdir(r"C:\computer vision\projects")

cap0=cv2.VideoCapture(0)
cap1=cv2.VideoCapture(r"C:\computer vision\projects\self driving car project\project_video.mp4",0)

apply_processing=False
#playsound("warning.mp3")
while True:

    ret0,frame0=cap0.read()
    ret1,frame1=cap1.read()
    frame0=cv2.resize(frame0,(640,480))
    frame1=cv2.resize(frame1,(640,480))
    h,w,_=frame1.shape
    
    if not apply_processing:    
        res=detect_drowsy(frame0)
        
    
    if apply_processing:
        #curve_estimation
        curve,frame1=getLane(frame1,w,h)
        #object_detection
        points=object_in_frame(frame1,1)
        cars=cars_in_roi(frame1,points)
        lable=object_in_frame(frame1,0)
        #traffic_signs_detection
        signs= traffic_signs(frame1,1)
        #traffic_lights_recognition
        lights=traffic_lights(frame1,1)
    
    cv2.imshow("drowsy face",frame0)
    cv2.imshow("result",frame1)
                
    key=cv2.waitKey(1)
    if key==ord("y"):
        os.system("ativation.mp3")
 #       ply("ativation.mp3")
        apply_processing=True
    elif key==ord("n"):
        os.system("desativation.mp3")
        #ply("desativation.mp3")
        apply_processing=False
    elif key==ord("q"):
        break 

cap0.release
cap1.release
cv2.destroyAllWindows()