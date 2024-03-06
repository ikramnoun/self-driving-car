import cv2
import numpy as np
from matplotlib import pyplot as plt

#img=cv2.imread(r"C:\computer vision\projects\self driving car project\cal_img.jpg")
#img=cv2.imread(r"C:\Users\Dell Latitude E5470\Pictures\2022-09-29\warped.jpg",1)
#img=cv2.resize(img,(640,480))
#img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#img_height,img_width=img.shape

def histogram(img):
    hist=np.sum(img[img.shape[0]//2:,:],axis=0)
    midpoint=np.int16(hist.shape[0]/2)  
    maxleft = np.max(hist[:midpoint])
    maxright = np.max(hist[midpoint:])
    
    indices_maxleft = np.where(hist[:midpoint] == maxleft)[0]
    indices_maxright = np.where(hist[midpoint:] == maxright)[0] + midpoint
    
    left_base=int(np.average(indices_maxleft))
    right_base=int(np.average(indices_maxright))
    
    #plt.plot(hist,color='blue')
    #plt.xlabel("column index")
    #plt.ylabel("sum of pixels intensities")

    #plt.scatter(midpoint, np.average(hist[midpoint]), color='red', marker='o', label='Midpoint')
    #plt.scatter(left_base, maxleft, color='green', marker='o', label='Left Base')
    #plt.scatter(right_base, maxright, color='purple', marker='o', label='Right Base')
    
    #plt.legend()
    #plt.show()

    return left_base,right_base

def sliding_window(img):
    left_base,right_base=histogram(img)
    
    nonzero=img.nonzero()
    nonzeroy=np.array(nonzero[0])
    nonzerox=np.array(nonzero[1])

    minpix=50
    left_lane_indices=[]
    right_lane_indices=[]

    
    windows=9
    margin=70
    img_height=img.shape[0]
    window_height=np.int16(img_height//windows)

    image=img.copy()
    image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

    for window in range(windows):
        window_y_low=img_height-window_height*(window+1)
        window_y_high=img_height-window_height*window
        window_xleft_low=left_base-margin
        window_xleft_high=left_base+margin
        window_xright_low=right_base-margin
        window_xright_high=right_base+margin
        
        cv2.rectangle(image,(window_xleft_low,window_y_low),(window_xleft_high,window_y_high),(0,255,0),3)
        cv2.rectangle(image,(window_xright_low,window_y_low),(window_xright_high,window_y_high),(0,255,0),3)

        good_left_indices=((nonzeroy>=window_y_low)&(nonzeroy<window_y_high)&(nonzerox>=window_xleft_low)&(nonzerox<window_xleft_high)).nonzero()[0]
        good_right_indices=((nonzeroy>=window_y_low)&(nonzeroy<window_y_high)&(nonzerox>=window_xright_low)&(nonzerox<window_xright_high)).nonzero()[0]
        
        if len(good_left_indices)>=minpix:
            left_base=np.int16(np.mean(nonzerox[good_left_indices]))
        if len(good_right_indices)>=minpix:
            right_base=np.int16(np.mean(nonzerox[good_right_indices]))
        
        left_lane_indices.append(good_left_indices)
        right_lane_indices.append(good_right_indices)
    
    try:
        left_lane_indices = np.concatenate(left_lane_indices)
        right_lane_indices= np.concatenate(right_lane_indices)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    leftx=nonzerox[left_lane_indices]
    rightx=nonzerox[right_lane_indices]
    lefty=nonzeroy[left_lane_indices]
    righty=nonzeroy[right_lane_indices]

    
    return image,leftx,lefty,rightx,righty        

def poly_fit(img,leftx,lefty,rightx,righty):
        
    left_coeff=np.polyfit(lefty,leftx,2)
    f=np.poly1d(left_coeff)
 #   print("f=",f)
    right_coeff=np.polyfit(righty,rightx,2)
    g=np.poly1d(right_coeff)
#    print("g=",g)

    ploty=np.linspace(0,img.shape[0]-1,img.shape[0])

    left_fitx=f(ploty)
    right_fitx=g(ploty)

    #plt.scatter(leftx,lefty,c="black")
    #plt.scatter(rightx,righty,c="black")
    #plt.plot(left_fitx,ploty,c="blue")
    #plt.plot(right_fitx,ploty,c="green")
    #plt.show()

    return left_fitx,right_fitx,ploty

def search_around_poly(img_warpped):

    margin=70

    nonzero=img_warpped.nonzero()
    nonzeroy=np.array(nonzero[0])
    nonzerox=np.array(nonzero[1])
    
    left_fitx = np.array([])
    right_fitx = np.array([])

    slides,leftx,lefty,rightx,righty=sliding_window(img_warpped)
    
    if (len(leftx)==0 or len(lefty)==0 or len(rightx)==0 or len(righty)==0):
        left_curverad=0
        right_curverad=0
        out_img = np.dstack((img_warpped, img_warpped, img_warpped)) * 255
    else:
        left_coeff=np.polyfit(lefty,leftx,2)
        f=np.poly1d(left_coeff)
        right_coeff=np.polyfit(righty,rightx,2)
        g=np.poly1d(right_coeff)

        left_x=f(nonzeroy)
        right_x=g(nonzeroy)
         
        left_lane_indices=((nonzerox > left_x-margin)&(nonzerox<left_x+margin))
        right_lane_indices=((nonzerox>right_x-margin)&(nonzerox<right_x+margin))

        leftx=nonzerox[left_lane_indices]
        rightx=nonzerox[right_lane_indices]
        lefty=nonzeroy[left_lane_indices]
        righty=nonzeroy[right_lane_indices]

        left_fitx,right_fitx,yy=poly_fit(img_warpped,leftx,lefty,rightx,righty) 
         
        ym_per_pixel=30/480
        xm_per_pixel=3.7/640

        left_fit_curve=np.polyfit(yy*ym_per_pixel,left_fitx*xm_per_pixel,2)
        right_fit_curve=np.polyfit(yy*ym_per_pixel,right_fitx*xm_per_pixel,2)
    
        y_eval = np.max(yy)*ym_per_pixel


        left_curverad = ((1 + (2 * left_fit_curve[0]*y_eval   + left_fit_curve[1]) ** 2) ** 1.5) / (2 * left_fit_curve[0])        
        right_curverad = ((1 + (2 * right_fit_curve[0]*y_eval   + right_fit_curve[1]) ** 2) ** 1.5) / (2 * right_fit_curve[0])

        
        slides[lefty,leftx]=[255,0,0]
        slides[righty,rightx]=[0,0,255]

        out_img=img_warpped.copy()
        
        out_img = np.dstack((img_warpped, img_warpped, img_warpped)) * 255

        left = np.array([np.transpose(np.vstack([left_fitx, yy]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yy])))])
        points = np.hstack((left, right))
        out_img = cv2.fillPoly(out_img, np.int_(points), (0, 255, 0))
        

    return slides,out_img,left_curverad,right_curverad ,left_fitx,right_fitx
    
 
        

#histogram(img)
#image,leftx,lefty,rightx,righty=sliding_window(img)
#left_fitx,right_fitx,ploty=poly_fit(img,leftx,lefty,rightx,righty)
#slids,out_img,left_curverad,right_curverad =search_around_poly(img)

#cv2.imshow("out_img",img)
#cv2.imshow("slides",slids)
#cv2.imshow("out_img",out_img)
#cv2.waitKey(0)