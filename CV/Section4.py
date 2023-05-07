import cv2
import numpy as np
face_detect=cv2.CascadeClassifier(r"D:\computer vision\face detection\haarcascade_frontalface_default.xml")
eye_detect=cv2.CascadeClassifier(r"D:\computer vision\face detection\haarcascade_eye.xml")#
smile_detect=cv2.CascadeClassifier(r"D:\computer vision\face detection\haarcascade_smile.xml")#
img=cv2.imread(r"D:\computer vision\h2.jpeg",1)
#detect faces 

#detect faces by haar cascadein image
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=face_detect.detectMultiScale(gray_img)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    img2=img[y:y+h,x:x+w]
#detect eyes in face
    eyes=eye_detect.detectMultiScale(img2)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(img2,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
#detect smiles in face
    smiles=smile_detect.detectMultiScale(img2,1.3,20) 
    for(sx,sy,sw,sh)in smiles: 
        cv2.rectangle(img2,(sx,sy),(sx+sw,sy+sh),(255,0,0),2)
cv2.imshow("livee",img)
cv2.waitKey(0)


#detect faces in video from camera lab
# stream = cv2.VideoCapture(0)

# if (stream.isOpened()== False): 
#   print("Error opening video stream or file")
  
# # Read until video is completed
# while(stream.isOpened()):
#     st,frame=stream.read()# for (x,y,w,h) in faces:

#     gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     faces=face_detect.detectMultiScale(gray_img,1.3,5)
# #detect faces from image    
#     for (x,y,w,h) in faces:
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
# #detect eyes in faces
#         img2=frame[y:y+h,x:x+w]
#         eyes=eye_detect.detectMultiScale(img2)
#         for (ex,ey,ew,eh) in eyes:
#             cv2.rectangle(img2,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
            
            
# #detect smile in faces 
#         smiles=smile_detect.detectMultiScale(img2,1.3,10)
#         for(sx,sy,sw,sh)in smiles:
#             cv2.rectangle(img2,(sx,sy),(sx+sw,sy+sh),(255,0,0),2)
#     cv2.imshow("livee",frame)
#     if cv2.waitKey(50)&0xff==ord("x"):
#         break
    
# stream.release()
# cv2.destroyAllWindows()

    
# # mouse event function
