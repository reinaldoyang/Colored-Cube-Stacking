import cv2 as cv
import numpy as np
from numpy.linalg import inv
import socket
import time
import serial


ser = serial.Serial(port='COM12', baudrate=115200, timeout=1, parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)

HOST = '192.168.0.1' # arm ip
PORT = 3000

#Load saved camera calibration data 
with np.load('CameraParameters.npz') as file:
    mtx,dist,rvecs,tvecs=[file[i] for i in ('cameraMatrix','dist','rvecs','tvecs')]


#capture video throguh webcam
webcam = cv.VideoCapture(0, cv.CAP_DSHOW)
# get vcap property 
width  = webcam.get(cv.CAP_PROP_FRAME_WIDTH)   # float `width`
height = webcam.get(cv.CAP_PROP_FRAME_HEIGHT)  # float `height`
print(width, height)

#for making the cv.imshow bigger
# webcam.set(3, 1920)
# webcam.set(4, 1080)

#width of the fov is 87 cm
# XPIXEL_TO_CM = 900/640
# YPIXEL_TO_CM = 630/480
XPIXEL_TO_CM = 900/640
YPIXEL_TO_CM = 630/480

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
            return cx_cm, cy_cm
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

def image_to_robot(image_coord, homogeneous_transformation):
    print("Object location (cm) in camera coordinate:" , image_coord)
    matrix = np.matmul(homogeneous_transformation, image_coord)
    matrix = matrix[0:2]
    print("Object location (cm) in robot base coordinate: ", matrix)
    return matrix

# homogeneous_transformation = [[-1, 0, 0, 680], 
#                             [0, 1, 0, 310], 
#                             [0, 0, -1, 846],
#                             [0, 0, 0, 1]]
homogeneous_transformation = [[-1, 0, 0, 680], 
                            [0, 1, 0, 240], 
                            [0, 0, -1, 846],
                            [0, 0, 0, 1]]

#define robot resting height
rest_height = 400

def send_coordinates(coordinates):
    s.send(bytes("{},{},{},{},{},{}".format(coordinates[0], coordinates[1], rest_height, 180, 0, 180), "utf-8"))
    time.sleep(3)
    # while True:  
    #     s.send(bytes("{},{},{},{},{},{}".format(coordinates[0], coordinates[1], rest_height, 180, 0, 180), "utf-8"))
    #     break

def rotateMatrixToEulerAngles2(RM):
    theta_z = np.arctan2(RM[1, 0], RM[0, 0]) / np.pi * 180
    theta_y = np.arctan2(-1 * RM[2, 0], np.sqrt(RM[2, 1] * RM[2, 1] + RM[2, 2] * RM[2, 2])) / np.pi * 180
    theta_x = np.arctan2(RM[2, 1], RM[2, 2]) / np.pi * 180
    # print(f"Euler angles:\ntheta_x: {theta_x}\ntheta_y: {theta_y}\ntheta_z: {theta_z}")
    print('Rx,Ry,Rz:\n',theta_x,theta_y,theta_z)
    return theta_x, theta_y, theta_z

    
def down(coordinates, z):

    s.send(bytes("{},{},{},{},{},{}".format(coordinates[0], coordinates[1], z, 180, 0, 180), "utf-8"))
    
def up(coordinates):
    sz = 400
    s.send(bytes("{},{},{},{},{},{}".format(coordinates[0], coordinates[1], sz, 180, 0, 180), "utf-8"))

def close():
    print("close")
    ser.write(serial.to_bytes([0x09,0x10,0x03,0xE8,0x00,0x03,0x06,0x09,0x00,0x00,0x6E,0x3F,0x3F,0x43,0x94]))

def open():
    print("open")
    ser.write(b'\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\x00\xFF\xFF\x72\x19')

place_coordinates = [500, 242]
def place(place_coordinates):
    # z = 167
    sz = 400
    s.send(bytes("{},{},{},{},{},{}".format(place_coordinates[0], place_coordinates[1], sz, 180, 0, 180), "utf-8"))
    # s.send(bytes("{},{},{},{},{},{}".format(place_coordinates[0], place_coordinates[1], z, 180, 0, 180), "utf-8"))
    # time.sleep(3)


while(True):
    ret, frame = webcam.read()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))

    # red(frame, lower_range_red, upper_range_red)
    # yellow(frame, lower_range_yellow, upper_range_yellow)
    # green(frame, lower_range_green, upper_range_green)
    # blue(frame, lower_range_blue, upper_range_blue)
    # orange(frame, lower_range_orange, upper_range_orange)

    #create if detected then retrieve coordinate
    if bool(red(frame, lower_range_red, upper_range_red)) == True:
        cx_red, cy_red = red(frame, lower_range_red, upper_range_red)

    img_coord = [cx_red, cy_red, 1, 1]
    
    cv.imshow('Video', frame)
    key = cv.waitKey(1)
    if key == ord('c'):
        print("Calculation:")
        location = image_to_robot(img_coord, homogeneous_transformation)
        continue

    if key == ord('s'):
        print("Sending coordinates")
        send_coordinates(location)
        time.sleep(3)
        down(location, 167)
        time.sleep(5)
        close()
        time.sleep(3)
        up(location)
        time.sleep(3)
        place(place_coordinates)
        time.sleep(5)
        down(place_coordinates, 172)
        time.sleep(5)
        open()
        time.sleep(1)
        up(place_coordinates)
        

        continue

    if key == 27: #Esc key
        break

    
#Release capture object after loop
webcam.release()
cv.destroyAllWindows