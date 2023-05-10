import cv2 as cv
import numpy as np
from numpy.linalg import inv

#Load saved camera calibration data 
with np.load('CameraParameters.npz') as file:
    mtx,dist,rvecs,tvecs=[file[i] for i in ('cameraMatrix','dist','rvecs','tvecs')]


#capture video throguh webcam
webcam = cv.VideoCapture(0, cv.CAP_DSHOW)

#for making the cv.imshow bigger
# webcam.set(3, 1920)
# webcam.set(4, 1080)

#width of the fov is 87 cm
XPIXEL_TO_CM = 60/1920
YPIXEL_TO_CM = 30/1080

lower_range_red = np.array([0, 177, 88])
upper_range_red = np.array([180, 255, 255])

#yellow values
lower_range_yellow = np.array([13, 111, 130])
upper_range_yellow = np.array([37, 255, 255])

#green values
lower_range_green = np.array([33, 71, 0])
upper_range_green = np.array([85, 255, 255])

#blue values
lower_range_blue = np.array([102, 99, 156])
upper_range_blue = np.array([179, 255, 255])

#orange values
lower_range_orange = np.array([0, 43, 149])
upper_range_orange = np.array([22, 255, 239])

def red(img, lower_range, upper_range):
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #define mask, cv.inrange is a method to detect colored object and returns binary image, if detected return white else black
    mask = cv.inRange(hsv_frame, lower_range, upper_range)
    _, mask1 = cv.threshold(mask, 244, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(mask1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for c in contours:
        x=600
        if cv.contourArea(c)>x:
            x,y,w,h=cv.boundingRect(c)
            M = cv.moments(c)
            # calculate x,y coordinate of centroid
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cx_cm = cx*XPIXEL_TO_CM
                cy_cm = cy*YPIXEL_TO_CM
                cv.circle(img, (cx, cy), 4, (255, 0, 255), -1)
                cv.putText(img, f'Red, centroid: ({cx}, {cy})', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv.putText(frame,("DETECT"),(10,60),cv.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
            return cx, cy
            # cv.putText(frame, 'red', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

def green(img, lower_range, upper_range):
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #define mask, cv.inrange is a method to detect colored object and returns binary image, if detected return white else black
    mask = cv.inRange(hsv_frame, lower_range, upper_range)
    _, mask1 = cv.threshold(mask, 244, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(mask1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for c in contours:
        x=600
        if cv.contourArea(c)>x:
            x,y,w,h=cv.boundingRect(c)
            M = cv.moments(c)
            # calculate x,y coordinate of centroid
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cx_cm = cx*XPIXEL_TO_CM
                cy_cm = cy*YPIXEL_TO_CM
                cv.circle(img, (cx, cy), 4, (255, 0, 255), -1)
                cv.putText(img, f'Green, centroid: ({cx}, {cy})', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            return cx, cy

def yellow(img, lower_range, upper_range):
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #define mask, cv.inrange is a method to detect colored object and returns binary image, if detected return white else black
    mask = cv.inRange(hsv_frame, lower_range, upper_range)
    _, mask1 = cv.threshold(mask, 244, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(mask1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for c in contours:
        x=600
        if cv.contourArea(c)>x:
            x,y,w,h=cv.boundingRect(c)
            M = cv.moments(c)
            # calculate x,y coordinate of centroid
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cx_cm = cx*XPIXEL_TO_CM
                cy_cm = cy*YPIXEL_TO_CM
                cv.circle(img, (cx, cy), 4, (255, 0, 255), -1)
                cv.putText(img, f'Yellow, centroid: ({cx}, {cy})', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            return cx, cy

def orange(img, lower_range, upper_range):
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #define mask, cv.inrange is a method to detect colored object and returns binary image, if detected return white else black
    mask = cv.inRange(hsv_frame, lower_range, upper_range)
    _, mask1 = cv.threshold(mask, 244, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(mask1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for c in contours:
        x=600
        if cv.contourArea(c)>x:
            x,y,w,h=cv.boundingRect(c)
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv.putText(frame,("DETECT"),(10,60),cv.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
            cv.putText(frame, 'orange', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

def blue(img, lower_range, upper_range):
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #define mask, cv.inrange is a method to detect colored object and returns binary image, if detected return white else black
    mask = cv.inRange(hsv_frame, lower_range, upper_range)
    _, mask1 = cv.threshold(mask, 244, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(mask1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for c in contours:
        x=600
        if cv.contourArea(c)>x:
            x,y,w,h=cv.boundingRect(c)
            M = cv.moments(c)
            # calculate x,y coordinate of centroid
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cx_cm = cx*XPIXEL_TO_CM
                cy_cm = cy*YPIXEL_TO_CM
                cv.circle(img, (cx, cy), 4, (255, 0, 255), -1)
                cv.putText(img, f'Yellow, centroid: ({cx}, {cy})', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            return cx, cy

def image_to_robot(image_coord, intrinsic_matrix, homogeneous_transformation):
    
    inverse_camera_matrix = inv(intrinsic_matrix)
    print("\nInverse Intrinsic Matrix: \n", inverse_camera_matrix)
    
    matrix = np.matmul(homogeneous_transformation, image_coord)
    # matrix = np.matmul(inverse_camera_matrix, image_coord)
    print("\n Multiplication of inverse camera matrix and image coord\n",matrix)

    #multiply the inverse camera matrix with z value (object to camera)
    zc=500
    matrix *= zc
    print("\n Multiply it by zc\n",matrix)

    # inverse_extrinsic_matrix = inv(extrinsic_matrix)
    matrix_2 = np.append(matrix, [1])

    print("\nAppend 0 to the end of the matrix \n", matrix_2)
    # print("\n Extrinsic_matrix: \n", extrinsic_matrix)
    # print("\n Inverse_extrinsic_matrix: \n", inverse_extrinsic_matrix)

    # matrix_2 = np.matmul(inverse_extrinsic_matrix, matrix_2)
    # print("\n Result \n",matrix_2)

# define the extrinsic matrix
extrinsic_matrix = [[-1, 0, 0, 266], 
                     [0, 1, 0, 576], 
                     [0, 0, -1, 920],
                     [0, 0, 0, 1]]

homogeneous_transformation = [[-1, 0, 0, 50], 
                                [0, 1, 0, 42], 
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]]


while(True):
    ret, frame = webcam.read()

    red(frame, lower_range_red, upper_range_red)
    yellow(frame, lower_range_yellow, upper_range_yellow)
    green(frame, lower_range_green, upper_range_green)
    blue(frame, lower_range_blue, upper_range_blue)
    # orange(frame, lower_range_orange, upper_range_orange)

    #create if detected then retrieve coordinate
    if bool(red(frame, lower_range_red, upper_range_red)) == True:
        cx_red, cy_red = red(frame, lower_range_red, upper_range_red)

    # # img_coord = [cx, cy, 1, 1]
    # cx_blue, cy_blue = blue(frame, lower_range_blue, upper_range_blue)
    cv.imshow('Video', frame)
    if cv.waitKey(1) == 27: #Esc key
        break
    if cv.waitKey(1) == ord('s'):
        print("successful")
    #     # print(img_coord)
    #     # image_to_robot(img_coord, mtx, homogeneous_transformation)
    
    
#Release capture object after loop
webcam.release()
cv.destroyAllWindows
