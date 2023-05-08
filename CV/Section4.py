"""
Created on 2023

@author: Hussein
"""

import cv2
import numpy as np

# Section 4

# Load the face, eye, and smile classifiers
face_detect = cv2.CascadeClassifier (r"Sec4\haarcascade_frontalface_default.xml")
eye_detect = cv2.CascadeClassifier (r"Sec4\haarcascade_eye.xml")
smile_detect = cv2.CascadeClassifier (r"Sec4\haarcascade_smile.xml")

img1 = cv2.imread(r"Sec4\image3.jpg",0 or 1 or -1)
img1 = cv2.resize(img1, (750, 700))


#detect faces 

'''
This code performs face detection on an image using the Haar Cascade classifier in OpenCV. 
The steps involved in this code are as follows:

1- Convert the original image img1 to grayscale using cv2.cvtColor() function.

2- Detect faces using the detectMultiScale() function of the face_detect classifier. 
   The function returns a list of rectangles where faces are detected, and the (x,y,w,h) tuple 
   represents the position of the detected face in the image.

3- For each detected face, draw a rectangle around it using the cv2.rectangle() 
   function on the original color image img1. The (x,y) coordinate is the top-left corner of the rectangle,
    and (x+w,y+h) is the bottom-right corner. The (0,255,0) is the color of the rectangle and 2 is the thickness.

4- Crop the detected face region from the original image using the (x,y,w,h) tuple and assign it to the img2 variable.

5- Detect eyes in the face region using the detectMultiScale() function of the eye_detect classifier,
   which returns a list of rectangles where eyes are detected. For each detected eye, 
   draw a rectangle around it using cv2.rectangle() on the cropped face region img2.

6- Detect smiles in the face region using the detectMultiScale() function of the smile_detect classifier,
   which returns a list of rectangles where smiles are detected. For each detected smile, 
   draw a rectangle around it using cv2.rectangle() on the cropped face region img2.

7- Display the final image with all the rectangles drawn around the detected faces, eyes, 
   and smiles using the cv2.imshow() function. cv2.waitKey(0) is used to wait for a key press before closing the window.

Overall, the code detects faces, eyes, and smiles in an image using Haar Cascade classifiers 
and highlights them with rectangles on the original image.
'''

#detect faces by haar cascadein image

# Convert the image to grayscale
gray_img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_detect.detectMultiScale(gray_img)

# For each detected face, draw a rectangle around it
for (x,y,w,h) in faces:
    cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),2)
    # Crop the detected face region from the original image
    img2 = img1[y:y+h,x:x+w]
    
    # Detect eyes in the face region and draw rectangles around them
    eyes = eye_detect.detectMultiScale(img2)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(img2,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
    
    # Detect smiles in the face region and draw rectangles around them
    smiles = smile_detect.detectMultiScale(img2,1.3,20) 
    for(sx,sy,sw,sh)in smiles: 
        cv2.rectangle(img2,(sx,sy),(sx+sw,sy+sh),(255,0,0),2)

# Display the final image with all the rectangles drawn around the detected faces, eyes, and smiles
cv2.imshow("livee",img1)
cv2.waitKey(0)


'''
This code uses the OpenCV library to detect faces, eyes, and smiles in a video stream captured from 
 the computer's default camera.

The code first creates a VideoCapture object called stream which initializes the default camera (device 0).

Then it checks if the camera is opened successfully using the isOpened() method. If the camera is not opened successfully,
 it prints an error message.

The code then enters a while loop which continues until the video stream is opened. Within the loop,
 it reads each frame from the video stream using the read() method of the stream object and converts the image 
 to grayscale using cv2.cvtColor() method. Then, the detectMultiScale() method of the face_detect object is used to
 detect the faces in the grayscale image.

After detecting the faces, the code draws a green rectangle around each face using the cv2.rectangle() method.

Then, for each face, the code creates a new image called img2 containing only that face. 
 The detectMultiScale() method of the eye_detect object is then used to detect the eyes in img2,
 and a red rectangle is drawn around each eye.

The detectMultiScale() method of the smile_detect object is used to detect the smiles in img2, 
 and a blue rectangle is drawn around each smile.

Finally, the modified image with rectangles around faces, eyes, and smiles is displayed in a 
 new window using cv2.imshow() method. The loop will continue until the user presses the "x" key, 
 upon which the program will terminate by breaking out of the loop.

Finally, the code releases the video stream and destroys all the windows created by the program using release()
 and destroyAllWindows() method.
'''

# Importing necessary libraries
import cv2

# Initializing the video capture object from the camera
stream = cv2.VideoCapture(0)

# Checking if the camera is available and working
if (stream.isOpened() == False): 
   print("Error opening video stream or file")

# Looping until the video is completed
while(stream.isOpened()):
    # Reading the frames from the camera
    st, frame = stream.read()
    
    # Converting the frame to grayscale for face detection
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detecting faces in the grayscale image
    faces = face_detect.detectMultiScale(gray_img, 1.3, 5)

    # Looping through all the detected faces
    for (x,y,w,h) in faces:
        # Drawing a green rectangle around the detected face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
        # Cropping the detected face region
        img2 = frame[y:y+h, x:x+w]
        
        # Detecting eyes in the cropped face region
        eyes = eye_detect.detectMultiScale(img2)
        
        # Looping through all the detected eyes
        for (ex, ey, ew, eh) in eyes:
            # Drawing a red rectangle around the detected eye
            cv2.rectangle(img2, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
            
        # Detecting smiles in the cropped face region
        smiles = smile_detect.detectMultiScale(img2, 1.3, 10)
        
        # Looping through all the detected smiles
        for(sx, sy, sw, sh) in smiles:
            # Drawing a blue rectangle around the detected smile
            cv2.rectangle(img2, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
    
    # Displaying the frame with the detected faces, eyes and smiles
    cv2.imshow("livee", frame)
    
    # Checking for the 'x' key press to exit the video capture
    if cv2.waitKey(50) & 0xff == ord("x"):
        break

# Releasing the video capture object and destroying all windows
stream.release()
cv2.destroyAllWindows()



