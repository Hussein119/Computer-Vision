import cv2
import numpy as np
face_detect=cv2.CascadeClassifier(r"D:\computer vision\face detection\haarcascade_frontalface_default.xml")
#eye_detect=cv2.CascadeClassifier(r"D:\computer vision\face detection\haarcascade_eye.xml")#
img=cv2.imread(r"D:\computer vision\h2.jpeg",1)
#detect faces 
# POINTS=[]
# def click_event(event, x, y, flags, param):
#       if event == cv2.EVENT_LBUTTONDOWN:
#           POINTS.append((x,y))
          
#       elif event == cv2.EVENT_LBUTTONUP:
#           POINTS.append((x,y))
          
#           if len(POINTS)==2:
#               cv2.rectangle(img,POINTS[0],POINTS[1],(0,255,0),2)
#               cv2.imshow("human", img)
#               POINTS.clear()
# cv2.imshow("human", img)
# cv2.setMouseCallback("human", click_event)
# cv2.waitKey(0)

# #detect faces by haar cascadein image
# gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# faces=face_detect.detectMultiScale(gray_img,1.3,5)
# print(faces)
# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# cv2.imshow("livee",img)
# cv2.waitKey(0)


#detect faces in video
stream = cv2.VideoCapture(r'C:\Users\Engohir\Downloads\Video\Humans TV Series Trailer.mp4')
if (stream.isOpened()== False): 
  print("Error opening video stream or file")
  
# Read until video is completed
while(stream.isOpened()):
    st,frame=stream.read()
    gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces=face_detect.detectMultiScale(gray_img,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("livee",frame)
    if cv2.waitKey(50)&0xff==ord("x"):
        break
stream.release()
cv2.destroyAllWindows()

    
