import cv2
import numpy as np

def nothing(a):
    pass

def trackbar_thres(initial_l,initial_u):
    cv2.namedWindow("trackbar_thres")
    cv2.createTrackbar("L_H","trackbar_thres",initial_l[0],255,nothing)
    cv2.createTrackbar("L_S","trackbar_thres",initial_l[1],255,nothing)
    cv2.createTrackbar("L_V","trackbar_thres",initial_l[2],255,nothing)
    cv2.createTrackbar("U_H","trackbar_thres",initial_u[0],255,nothing)
    cv2.createTrackbar("U_S","trackbar_thres",initial_u[1],255,nothing)
    cv2.createTrackbar("U_V","trackbar_thres",initial_u[2],255,nothing)

def thresholding(img):
    l_h=cv2.getTrackbarPos("L_H","trackbar_thres")
    l_s=cv2.getTrackbarPos("L_S","trackbar_thres")
    l_v=cv2.getTrackbarPos("L_V","trackbar_thres")
    u_h=cv2.getTrackbarPos("U_H","trackbar_thres")
    u_s=cv2.getTrackbarPos("U_S","trackbar_thres")
    u_v=cv2.getTrackbarPos("U_V","trackbar_thres")

    lower=np.array([l_h,l_s,l_v])
    upper=np.array([u_h,u_s,u_v])

    hsv_frame=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    thres=cv2.inRange(hsv_frame,lower,upper)

    return thres


def trackbar_warp(inistial,w,h):
    cv2.namedWindow("trackbar_warp")
    cv2.resizeWindow('trackbar_warp',(360,250))
    cv2.createTrackbar("w_T","trackbar_warp",inistial[0],w,nothing)
    cv2.createTrackbar("h_T","trackbar_warp",inistial[1],h,nothing)
    cv2.createTrackbar("w_B","trackbar_warp",inistial[2],w,nothing)
    cv2.createTrackbar("h_B","trackbar_warp",inistial[3],h,nothing)

def valtrackbars(w=640,h=480):
    width_top=cv2.getTrackbarPos("w_T","trackbar_warp")
    height_top=cv2.getTrackbarPos("h_T","trackbar_warp")
    width_bottom=cv2.getTrackbarPos("w_B","trackbar_warp")
    height_bottom=cv2.getTrackbarPos("h_B","trackbar_warp")  
    points=np.float32([(width_top-20,height_top),(w-width_top-20,height_top),(width_bottom-20,height_bottom),(w-width_bottom-20,height_bottom)])
    return points

def drawpoints(points,img):
    copy=img.copy()
    for x in range(4):
        cv2.circle(copy,(int(points[x][0]),int(points[x][1])),4,(0,0,255),-1)
    return copy


def warp_img(img,points,w,h,inv=False):
    pts1=np.float32(points)
    pts2=np.float32([[0,0],[w,0],[0,h],[w,h]])
    if inv:
        matrix=cv2.getPerspectiveTransform(pts2,pts1)
    else:
      matrix=cv2.getPerspectiveTransform(pts1,pts2)

    warp=cv2.warpPerspective(img,matrix,(w,h))

    return warp

def drawlines(img_warp):
    warp_copy=img_warp.copy()
    warp_copy=cv2.cvtColor(warp_copy,cv2.COLOR_GRAY2BGR)
    
    edges=cv2.Canny(img_warp,75,150)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength=50,maxLineGap=20)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line[0]
            cv2.line(warp_copy,(x1,y1),(x2,y2),(255,0,0),30)
    return warp_copy


#img=cv2.imread(r"C:\Users\Dell Latitude E5470\Pictures\2022-09-29\Screenshot 2024-02-24 002028.jpg",1)
#img=cv2.resize(img,(720,640))
#h,w,c=img.shape

#inistial_l=[0,0,205]
#inistial_u=[255,255,255]
#trackbar_thres(inistial_l,inistial_u)

#inistialVal=[220,480,120,600]
#trackbar_warp(inistialVal,w=720,h=640)

#thres=thresholding(img)
#points=valtrackbars(w,h)
#roi=drawpoints(points,img)
#img_warp=warp_img(thres,points,w,h)
#lines=drawlines(img_warp)

#cv2.imshow("img",img)
#cv2.imshow("threshold",thres)
#cv2.imshow("ROI",roi)
#cv2.imshow('img_warp',img_warp)
#cv2.imshow("lines",lines)
#cv2.waitKey(0)
#while(True):
 #   points=valtrackbars(w,h)
  #  roi=drawpoints(points,img)
   # cv2.imshow("ROI",roi)
    #if cv2.waitKey(1)=="q":
     #   break
