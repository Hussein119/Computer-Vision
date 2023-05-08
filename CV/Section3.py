"""
Created on 2023

@author: Hussein
"""
import cv2
import numpy as np
'''
Open a terminal or command prompt.
Type the following command and press Enter: pip install matplotlib
Wait for the installation to complete.
'''
import matplotlib.pyplot as plt 


# Section 3

img1 = cv2.imread(r"Sec3\image.jpg",0 or 1 or -1)

# Calling  Split and  Merge from Section2

# Split

b, g, r = cv2.split(img1)
cv2.imshow('Blue', b)
cv2.waitKey(0)
cv2.imshow('Green', g)
cv2.waitKey(0)
cv2.imshow('Red', r)
cv2.waitKey(0)
cv2.imwrite(r'Sec2\b.jpg', b)
cv2.imwrite(r'Sec2\g.jpg', g)
cv2.imwrite(r'Sec2\r.jpg', r)

# Merge

# Merge three channels (blue, green, and red) into a single RGB image
image_merge = cv2.merge([b, g, r])

# Display the merged image
cv2.imshow("RGB_Image", image_merge)

# Save the merged image to disk
cv2.imwrite(r'Sec2\image_merge.jpg', image_merge)

# Wait for a key event and exit when any key is pressed
cv2.waitKey(0)

# Create a black square image with a size of 400x400 pixels
square = np.zeros((400,400),dtype="uint8")

# Display the square image
cv2.imshow("square",square)

# Wait for a key event and exit when any key is pressed
cv2.waitKey(0)

# Draw a white rectangle in the center of the square image
img_rec = cv2.rectangle(square,(50,50),(350,350),255,-1)

# Display the image with the rectangle
cv2.imshow("img_rec",img_rec)

# Wait for a key event and exit when any key is pressed
cv2.waitKey(0)

# Create a black circle image with a size of 400x400 pixels
circle = np.zeros((400,400),dtype="uint8")

# Draw a white circle in the center of the circle image
img_cir = cv2.circle(circle,(200,200),200,255,-1)

# Display the image with the circle
cv2.imshow("circle",img_cir)

# Wait for a key event and exit when any key is pressed
cv2.waitKey(0)


# And operation 
and_img = cv2.bitwise_and(img_rec, img_cir, mask = None)
cv2.imshow("AND", and_img)
cv2.waitKey(0)

#Or operation
or_img = cv2.bitwise_or(img_rec, img_cir)
cv2.imshow("OR", or_img)
cv2.waitKey(0)

#xor operation
xor_img = cv2.bitwise_xor(img_rec, img_cir)
cv2.imshow("xor", xor_img)
cv2.waitKey(0)

#Not operation
not_img = cv2.bitwise_not(img_cir)
cv2.imshow("NOT", not_img)
cv2.waitKey(0)

# Image masking

# create a black (all-zero) image with the same shape as input image
mask = np.zeros(img1.shape[:2], dtype=np.uint8) # 8-bit unsigned integer (0 to 255)

# find the center of input image
center = (img1.shape[1] // 2, img1.shape[0] // 2) 

# draw a white circle on the black image
c = cv2.circle(mask, center, 200, 255, -1)

# show the masked image
cv2.imshow("Mask", mask)

# wait for a key event to exit
cv2.waitKey(0)


# Perform bitwise AND operation between img1 and mask using cv2.bitwise_and() function
# This operation creates a new image (img_msk) where only the pixels that correspond to 1 (white) in the mask image are kept
img_msk = cv2.bitwise_and(img1, img1, mask=mask)

# Display the masked image using cv2.imshow() function
cv2.imshow("masked image", img_msk)

# Wait until a key is pressed to close the window
cv2.waitKey(0)

# mouse event function

# Define a function called click_event that will be called when the mouse is clicked
def click_event(event, x, y, flags, param):
    
    # If the left mouse button is clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        
        # Print the (x, y) coordinates of the mouse click on the console
        print(x, ', ', y)
        
        # Create a string of the (x, y) coordinates
        strxy = str(x) + ', ' + str(y)
        
        # Draw the (x, y) coordinates on the image using cv2.putText() function
        cv2.putText(img1, strxy, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
        
        # Show the image with the text using cv2.imshow() function
        cv2.imshow("Michky", img1)

    # If the right mouse button is clicked
    if event == cv2.EVENT_RBUTTONDOWN:
        
        # Get the (B, G, R) values of the pixel at the (x, y) coordinates
        b = img1[y, x, 0]
        g = img1[y, x, 1]
        r = img1[y, x, 2]
        
        # Create a string of the (B, G, R) values
        strbgr = str(b) + ', ' + str(g) + ', ' + str(r)
        
        # Draw the (B, G, R) values on the image using cv2.putText() function
        cv2.putText(img1, strbgr, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
        
        # Show the image with the text using cv2.imshow() function
        cv2.imshow("Michky", img1)



# Display the image using cv2.imshow() function
cv2.imshow("Michky", img1)

# Set the mouse callback function using cv2.setMouseCallback() function
cv2.setMouseCallback("Michky", click_event)

# Wait until a key is pressed to close the window
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()


'''
This code is using two different libraries, matplotlib and OpenCV, to read and display an image,
 and to calculate and plot its histogram.

First, the plt.imread() function from matplotlib is used to read an image file named 
 "image.jpg" located in the "Sec3" directory. The returned image is stored in a variable named img_mpl. 

The prefix "r" before the path name is used to indicate that the string should be treated as a "raw string",
 meaning that backslashes are treated as literal characters and not escape characters.

Then, the cv2.imshow() function from OpenCV is used to display the image img_mpl in a window with the title "asd".

Next, the cv2.calcHist() function from OpenCV is used to calculate the histogram of the image img1,
 which is not defined in the code snippet provided. The histogram is calculated for the red channel ([2] argument), 
 with 256 bins covering the range of values between 0 and 255.

After that, the plt.plot() function from matplotlib is used to plot the histogram calculated in the previous step.

Finally, the cv2.waitKey() function is used to wait for a key press to close the window displaying the image.
'''
# Matplotlib
# Read the image using Matplotlib
img_mpl = plt.imread(r"Sec3\image.jpg")

# Display the image using OpenCV
cv2.imshow("asd", img_mpl)

# Calculate the histogram of the blue channel (index 2) using OpenCV
histr = cv2.calcHist([img1], [2], None, [256], [0, 255])

# Plot the histogram
plt.plot(histr)

# Display the image using Matplotlib
plt.imshow(img1)

# Show the plot
plt.show()

# Wait for a key event
cv2.waitKey(0)
