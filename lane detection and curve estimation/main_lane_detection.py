import cv2
import numpy as np
from utils_lane_detection import *
from lane_lines_detection import *

inistial_l=[0,0,200]
inistial_u=[255,255,255]
trackbar_thres(inistial_l,inistial_u)

inistialVal=[245,305,120,395]
#inistialVal=[220,450,100,560]
trackbar_warp(inistialVal,w=640,h=480)

cap=cv2.VideoCapture(r"self driving car project/project_video.mp4",0)
#cap=cv2.VideoCapture(r"C:\computer vision\projects\self driving car project\project_video1.mp4",0)
#cap=cv2.VideoCapture(r"C:\Users\Dell Latitude E5470\Pictures\lv_0_٢٠٢٤٠٢٢٣٢٢٢٦٥٩.mp4")
def getLane(img,w,h):
      
      
        thres=thresholding(img)
        
        points=valtrackbars(w,h)
        roi=drawpoints(points,img)
        img_warp=warp_img(thres,points,w,h)

        lines=drawlines(img_warp)
    
        slides,out,left_curverad,right_curverad,left_fitx,right_fitx =search_around_poly(img_warp)
        
        
        out_img=cv2.addWeighted(out,1,lines,1,0)
          
        inv_img_warp=warp_img(out_img,points,w,h,inv=True)
        result=cv2.addWeighted(img,1,inv_img_warp,1,0)
        if len(left_fitx) > 0 and len(right_fitx) > 0:
          curvature=(left_curverad+right_curverad)/2
          cv2.circle(out_img,(int(curvature),int(img_warp.shape[0]/2)),10,(0,0,255),-1)
          car_pos=img.shape[1]/2
          cv2.circle(out_img,(int(car_pos),int(img_warp.shape[0])),10,(255,0,0),-1)
          lane_center=np.average((left_fitx[-1] + right_fitx[-1])/2)
          cv2.circle(out_img,(int(lane_center),int(img_warp.shape[0])),10,(0,0,255),-1)
          center=abs((car_pos-lane_center)*(3.7/640))

          text1="Raduis of curvature:"+str(round(curvature,2))+"m"
          cv2.putText(result,text1,(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

          if abs(curvature)>5000:
              text2="go strigh"
          elif curvature<0:
              text2="turn left:"+str((round(center,3)))+"m away from the center"
          elif curvature>0 and curvature<5000:
              text2="turn right:"+str((round(center,3)))+"m away from the center"      

          cv2.putText(result,text2,(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
        else:
            center=0
            cv2.putText(result,"No lane lines detected",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2) 
       # cv2.imshow("img",img)
      #  cv2.imshow("threshold",thres)
      #  cv2.imshow("roi",roi)
        cv2.imshow('img_warp',img_warp)
      #  cv2.imshow("lines",lines)
        cv2.imshow("out_img",out_img)
        cv2.imshow("slides",slides)
        cv2.imshow("inv_img_warp",inv_img_warp)
        cv2.imshow("result",result)
        return car_pos,result
frameCounter = 0
while True:
    frameCounter +=1
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) ==frameCounter:
      cap.set(cv2.CAP_PROP_POS_FRAMES,0)
      frameCounter=0

    ret,frame=cap.read()

    frame=cv2.resize(frame,(640,480))
    h,w,_=frame.shape

    if ret==True:
     
        getLane(frame,w,h)

        if cv2.waitKey(30)==ord("q"):
            break
    else:
        break
cap.release
cv2.destroyAllWindows()