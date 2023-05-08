"""
Created on 2023

@author: Hussein
"""

import cv2
import numpy as np


# Section 5

# Load the face, eye, and smile classifiers
face_detect = cv2.CascadeClassifier (r"Sec5\haarcascade_frontalface_default.xml")
eye_detect = cv2.CascadeClassifier (r"Sec5\haarcascade_eye.xml")
smile_detect = cv2.CascadeClassifier (r"Sec5\haarcascade_smile.xml")


img1 = cv2.imread(r"Sec4\image3.jpg",0 or 1 or -1)
img1 = cv2.resize(img1, (750, 700))


'''
This code allows the user to select a rectangular region of interest (ROI) in an image by clicking and dragging the mouse.
 Here's a breakdown of how it works:

The POINTS variable is initialized as an empty list to store the clicked points.
The click_event() function is defined to handle mouse events, which are triggered 
 when the user clicks or releases the mouse button. It takes in five arguments:
event: the type of mouse event (e.g. cv2.EVENT_LBUTTONDOWN for left button down)
x: the x-coordinate of the mouse cursor
y: the y-coordinate of the mouse cursor
flags: any modifier keys that were pressed during the mouse event
param: any extra parameters that were passed to setMouseCallback()
In the click_event() function:
If the left mouse button is clicked down (cv2.EVENT_LBUTTONDOWN), the (x,y) coordinates of
 the click are appended to the POINTS list.
If the left mouse button is released (cv2.EVENT_LBUTTONUP), the (x,y) coordinates of 
 the release are also appended to the POINTS list.
If there are exactly two points in the POINTS list, a green rectangle is drawn on
 the img1 image using the two points as opposite corners, with a thickness of 2 pixels. 
  The resulting image is then displayed using cv2.imshow().
Finally, the POINTS list is cleared to allow the user to select another rectangle.
The img1 image is displayed using cv2.imshow() before setting the mouse callback function 
 using cv2.setMouseCallback(). This sets up the window to listen for mouse events and call the click_event() 
  function when an event occurs.
The program waits for a key press using cv2.waitKey(0) before exiting.
Overall, this code provides a simple way for users to select rectangular regions of interest in an image, 
 which can be useful for various computer vision applications such as object detection or image segmentation.
'''
# Initialize an empty list to store clicked points
POINTS=[]

# Define the mouse click event handler function
def click_event(event, x, y, flags, param):
    # If the left mouse button is clicked down, append the coordinates to the list of points
    if event == cv2.EVENT_LBUTTONDOWN:
        POINTS.append((x,y))
    # If the left mouse button is released, append the coordinates and draw a rectangle
    elif event == cv2.EVENT_LBUTTONUP:
        POINTS.append((x,y))
        # If exactly two points have been clicked, draw a rectangle using the two points as opposite corners
        if len(POINTS)==2:
            cv2.rectangle(img1, POINTS[0], POINTS[1], (0,255,0), 2)
            cv2.imshow("human", img1)
            # Clear the list of clicked points to allow the user to select another rectangle
            POINTS.clear()

# Display the image to be annotated
cv2.imshow("human", img1)

# Set up the mouse callback function to listen for mouse events in the "human" window
cv2.setMouseCallback("human", click_event)

# Wait for a key press (0 means wait indefinitely) before closing the window and exiting the program
cv2.waitKey(0)


#detect faces by haar cascadein image
# Convert the input image to grayscale using the cv2.cvtColor() function
gray_img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image using the face_detect object (a Haar cascade classifier) 
# with a scale factor of 1.3 and a minimum neighbor count of 5.
faces = face_detect.detectMultiScale(gray_img,1.3,5)

# Print the bounding box coordinates of the detected faces
print(faces)

# For each face detected, draw a green rectangle around it on the original color image (img1) 
# using the cv2.rectangle() function with a thickness of 2 pixels
for (x,y,w,h) in faces:
     cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),2)

# Display the annotated image in a window titled "livee" using cv2.imshow() 
# and wait for a key press to close the window using cv2.waitKey(0)
cv2.imshow("livee",img1)
cv2.waitKey(0)



#detect faces in video
# Open the video file using OpenCV's VideoCapture object
stream = cv2.VideoCapture(r'Sec5\Humans TV Series Trailer.mp4')

# Check if the video file was successfully opened
if (stream.isOpened()== False): 
  print("Error opening video stream or file")
  
# Loop through the video frames until the end of the video is reached
while(stream.isOpened()):
    # Read the next frame from the video stream
    st,frame = stream.read()
    
    # Convert the frame to grayscale
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image using the face_detect cascade classifier
    faces = face_detect.detectMultiScale(gray_img,1.3,5)

    # Draw a green rectangle around each detected face
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    # Display the current frame with the detected faces
    cv2.imshow("livee",frame)
    
    # Wait for a key press for 50 milliseconds and check if the 'x' key was pressed to exit the loop
    if cv2.waitKey(50)&0xff==ord("x"):
        break

# Release the video stream object and close all windows
stream.release()
cv2.destroyAllWindows()


    
