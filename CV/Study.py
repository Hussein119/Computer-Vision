from ctypes.wintypes import POINT
import cv2 

import numpy as np

import matplotlib.pyplot as  plt

# Load the face, eye, and smile classifiers
face_detect = cv2.CascadeClassifier (r"Sec4\haarcascade_frontalface_default.xml")
eye_detect = cv2.CascadeClassifier (r"Sec4\haarcascade_eye.xml")
smile_detect = cv2.CascadeClassifier (r"Sec4\haarcascade_smile.xml")

# read a image
img = cv2.imread("resized_img.PNG" , 0 or 1 or -1)


stream = cv2.VideoCapture(0)

if(stream.isOpened() == False):
    print ("error")

while(stream.isOpened()):
     st , img =  stream.read()
     gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
     faces = face_detect.detectMultiScale(gray_img)
     for (x,y,h,w) in faces:
          cv2.rectangle(img,(x,y),(x+h,y+w),(255,0,0),2)
          img2 = img[y:y+h,x:x+w]
          eyes = eye_detect.detectMultiScale(img2)
          for (xw,yw,hw,ww) in eyes:
              cv2.rectangle(img2,(xw,yw),(xw+hw,yw+ww),(255,0,255),2)
          smile = smile_detect.detectMultiScale(img2,1.3,20)
          for (xs,ys,hs,ws) in smile:
              cv2.rectangle(img2,(xs,ys),(xs+hs,ys+ws),(0,255,255),2)
          
     cv2.imshow("Image",img)
     if cv2.waitKey(50) & 0xff == ord("x"):
              break
stream.release()
cv2.destroyAllWindows()


stream = cv2.VideoCapture(r'Sec5\Humans TV Series Trailer.mp4')

if(stream.isOpened() == False):
    print ("error")

while(stream.isOpened()):
     st , img =  stream.read()
     gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
     faces = face_detect.detectMultiScale(gray_img)
     for (x,y,h,w) in faces:
          cv2.rectangle(img,(x,y),(x+h,y+w),(255,0,0),2)
          img2 = img[y:y+h,x:x+w]
          eyes = eye_detect.detectMultiScale(img2)
          for (xw,yw,hw,ww) in eyes:
              cv2.rectangle(img2,(xw,yw),(xw+hw,yw+ww),(255,0,255),2)
          smile = smile_detect.detectMultiScale(img2,1.3,20)
          for (xs,ys,hs,ws) in smile:
              cv2.rectangle(img2,(xs,ys),(xs+hs,ys+ws),(0,255,255),2)
          
     cv2.imshow("Image",img)
     if cv2.waitKey(50) & 0xff == ord("x"):
              break
stream.release()
cv2.destroyAllWindows()


points=[]
def click (event,x,y,falgs,param):
    if(event == cv2.EVENT_RBUTTONDOWN):
        points.append((x,y))
    elif(event == cv2.EVENT_RBUTTONUP):
        points.append((x,y))
        if len(points) == 2:
            cv2.rectangle(img,points[0],points[1],(0,0,255),-1)
            cv2.imshow("img",img)
            points.clear()

cv2.imshow("img",img)
cv2.setMouseCallback("img",click)
     





