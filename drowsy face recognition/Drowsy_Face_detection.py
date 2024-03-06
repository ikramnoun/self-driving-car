import cv2
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
import dlib
#from playsound import playsound
import os


def eye_aspect_ratio(eye):
    A=distance.euclidean(eye[1],eye[5])
    B=distance.euclidean(eye[2],eye[4])
    C=distance.euclidean(eye[0],eye[3])
    ear=(A+B)/(2*C)
    return ear

    
detect=dlib.get_frontal_face_detector()
predict=dlib.shape_predictor(r"C:\computer vision\projects\recurements\shape_predictor_68_face_landmarks.dat")
(lstart,lend)=face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rstart,rend)=face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
thres=0.23
frame_check=60

flags=[]
def detect_drowsy(frame):       
    #while True:
        #ret,frame=cap.read()
        flag=0
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        subjects=detect(gray,0)
        for subject in subjects:
            
            shape=predict(gray,subject)
            shape=face_utils.shape_to_np(shape)
            
            left_eye=shape[lstart:lend]
            right_eye=shape[rstart:rend]
            
            left_ear=eye_aspect_ratio(left_eye)
            right_ear=eye_aspect_ratio(right_eye)
            ear=(left_ear+right_ear)/2
            
            left_eye_hull=cv2.convexHull(left_eye)
            right_eye_hull=cv2.convexHull(right_eye)

            cv2.drawContours(frame,[left_eye_hull],-1,(0,255,0),1)
            cv2.drawContours(frame,[right_eye_hull],-1,(0,255,0),1)

            if ear<thres:
                flag+=1
                flags.append(flag)
                cv2.putText(frame,"Flag:"+str(len(flags)),(30,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
                if len(flags)==frame_check:
                    #os.system("warning.mp3")
                    #playsound("warning.mp3")
                    print("you are drowsy")
                
                elif len(flags)>frame_check:
                    cv2.putText(frame,'""""""" you are DROWSY !!! """""""',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                    
                    
            else:
                flag=0
                flags.clear()
        return frame


cap=cv2.VideoCapture(0)                
while True:
    ret,img=cap.read()
    img=cv2.resize(img,(720,640))
    if ret==True:
        res=detect_drowsy(img)
        cv2.imshow("res",res)
        if cv2.waitKey(1)==ord("q"):
            break
    else:
        break
cap.release
cv2.destroyAllWindows()

