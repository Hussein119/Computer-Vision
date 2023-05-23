import cv2

# Question 1
# Write a code snippet to load and display an image named "image.jpg" using OpenCV.
image = cv2.imread("image.jpg")
cv2.imshow("Image", image)
cv2.waitKey(0)

# Question 2
# Modify the code to resize the image to a width of 800 pixels and maintain the aspect ratio.
resized_image = cv2.resize(image, (800, int(image.shape[0] * 800 / image.shape[1])))
cv2.imshow("Resized Image", resized_image)
cv2.waitKey(0)

# Question 3
# Implement face detection in the resized image using the Haar Cascade classifier for faces.
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

# Question 4
# Print the number of faces detected in the image.
print("Number of faces detected:", len(faces))

# Question 5
# For each detected face, draw a rectangle around it using a blue color and a thickness of 2 pixels.
for (x, y, w, h) in faces:
    cv2.rectangle(resized_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow("Annotated Image", resized_image)
cv2.waitKey(0)

# Question 6
# Save the annotated image as "annotated_image.jpg".
cv2.imwrite("annotated_image.jpg", resized_image)

# Question 7
# Implement eye detection within each detected face region using the Haar Cascade classifier for eyes.
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

for (x, y, w, h) in faces:
    face_region = resized_image[y:y + h, x:x + w]
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 3)

    # Question 8
    # Print the number of eyes detected in each face.
    print("Number of eyes detected in a face:", len(eyes))

    # Question 9
    # For each detected eye, draw a rectangle around it using a red color and a thickness of 2 pixels.
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(face_region, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

    cv2.imshow("Face with Eyes", resized_image)
    cv2.waitKey(0)

# Question 10
# Save the final annotated image with face and eye rectangles as "final_image.jpg".
cv2.imwrite("final_image.jpg", resized_image)

# Question 11
# Write a code snippet to open a video file named "Humans TV Series Trailer.mp4" using OpenCV.
video = cv2.VideoCapture("Humans TV Series Trailer.mp4")

# Question 12
# Implement real-time face detection in the video stream using the Haar Cascade classifier for faces.
while True:
    ret, frame = video.read()

    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    # Question 13
    # For each detected face, draw a rectangle around it using a green color and a thickness of 2 pixels.
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Question 14
    # Display the annotated video stream.
    cv2.imshow("Annotated Video", frame)

    # Question 15
    # Wait for the 'q' key to be pressed to exit the program.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream object and close all windows
video.release()
cv2.destroyAllWindows()
