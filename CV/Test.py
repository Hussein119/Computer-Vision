import cv2

#'''
#Image Height       :  2802
#Image Width        :  4195
#'''

#image = cv2.imread(r"C:\Users\Hos10\OneDrive\Pictures\Screenshots\Screenshot (194).png",0 or 1 or -1)

#resized_image = cv2.resize(image, (4195, 2802))

#cv2.imwrite("imgae.PNG",resized_image)

#face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#video = cv2.VideoCapture("Humans TV Series Trailer.mp4")

#while True :
#    ret , fram = video.read()
#    if not ret :
#        break
#    gray_fram = cv2.cvtColor(fram,cv2.COLOR_BGR2GRAY)
#    faces = face_cascade.detectMultiScale(gray_fram,1.3,5)
#    for (x,y,w,h) in faces : 
#        cv2.rectangle(fram,(x,y),(x+w,y+h),(0,255,0),2)
#    cv2.imshow("Video" , fram)
#    if(cv2.waitKey(1) & 0xFF == ord('q')):
#        break
#video.release()

import cv2

# Load the face, eye, and smile classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Function to detect and draw rectangles around faces, eyes, and smiles
def detect_faces_eyes_smiles(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # For each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract the region of interest (ROI) for face detection
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        
        # Detect eyes in the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
        
        # For each detected eye
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around the eye (within the face ROI)
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
        # Detect smiles in the face ROI
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20)
        
        # For each detected smile
        for (sx, sy, sw, sh) in smiles:
            # Draw a rectangle around the smile (within the face ROI)
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    
    # Display the image with rectangles around faces, eyes, and smiles
    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load an image and perform face, eye, and smile detection
image_path = 'resized_img.PNG'  # Replace with the path to your image
image = cv2.imread(image_path)
detect_faces_eyes_smiles(image)

# Perform face, eye, and smile detection in a live video stream
video_path = 'Humans TV Series Trailer.mp4'  # Replace with the path to your video file or 0 for webcam
video_capture = cv2.VideoCapture(video_path)

while True:
    # Read a frame from the video stream
    ret, frame = video_capture.read()
    
    # If the frame was not successfully read, exit the loop
    if not ret:
        break
    
    # Perform face, eye, and smile detection on the frame
    detect_faces_eyes_smiles(frame)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
video_capture.release()
cv2.destroyAllWindows()



