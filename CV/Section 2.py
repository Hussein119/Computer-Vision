"""
Created on 2023

@author: Hussein
"""

import cv2
import numpy as np


# Section 2 

# CONETNT :
# - Arithmetic operations
# - Make border
# - Crop image
# - Drawing on image
# - Write text
# - Image transformation: translation / Rotate / flip
# - Splitting & Merging


#  add two pixel in cv2

'''
This line adds two grayscale pixel values of 200 and 100 using the cv2.add() function.
Since the pixel values are represented as 8-bit unsigned integers, 
 we need to cast the values to the appropriate data type using np.uint8(). 
The result of the operation is a pixel value of 255, which is the maximum possible value for an 8-bit unsigned integer. 
This pixel value is printed to the console as part of a formatted string.
'''
# Add two grayscale pixel values
result = cv2.add(np.uint8([200]), np.uint8([100]))

# Print the result
print("MAX of 255: {}".format(result))

'''
This line subtracts two grayscale pixel values of 100 and 200 using the cv2.subtract() function.
Again, we need to cast the pixel values to np.uint8() before performing the operation.
Since the result of the operation is negative, the resulting pixel value is clamped to 0,
 which is the minimum possible value for an 8-bit unsigned integer. 
This pixel value is printed to the console as part of a formatted string.
'''
print("min of 255: {}".format(cv2.subtract(np.uint8([100]),np.uint8([200]))))
'''
The cv2.add() function adds two pixel values and saturates the result at the maximum value of 255,
while the cv2.subtract() function subtracts two pixel values and saturates the result at the minimum value of 0.
'''



#    add two pixel in numpy
print("wrap around: {}".format(np.uint8([200]) + np.uint8([100])))
print("wrap around: {}".format(np.uint8([100]) - np.uint8([200])))


# height, width, number of channels in image  
img1 = cv2.imread(r"Sec2\image.jpg",0 or 1 or -1)
height = img1.shape[0]  
width = img1.shape[1]  
channels = img1.shape[2]  
size1 = img1.size  

print('Image Dimension')  
print('Image Height       : ',height)  
print('Image Width        : ',width)  
print('Number of Channels : ',channels)  
print('Image Size  :', size1) 

#   add two image 
img2 = cv2.imread(r"Sec2\image.jpg",0 or 1 or -1)
'''
This line creates a NumPy array of the same shape as img2 using the np.ones() function.
Each element of the array is initialized to 1, and the data type is set to unsigned 8-bit integer
 using the dtype parameter. 
The resulting array is then multiplied by 100 to create an array M that contains all 100 values.
'''
M1 = np.ones(img2.shape,dtype="uint8") * 100
#print(M)
'''
This line performs pixel-wise addition between img2 and M using the cv2.add() function. 
Since M contains all 100 values, this operation effectively brightens the image by 100 intensity levels.
The result is stored in a new variable named added.
'''
added = cv2.add(img2,M1)

#Finally, this line displays the resulting image added in a window with the title "added" using the cv2.imshow() function.
cv2.imshow("added",added)
cv2.imwrite(r'Sec2\added_image.jpg', added)
cv2.waitKey(0)


#subtract two image 
img3 = cv2.imread(r"Sec2\image.jpg",0 or 1 or -1)
M2 = np.ones(img3.shape,dtype="uint8")*50
sub = cv2.subtract(img3,M2)
cv2.imshow("sub",sub)
cv2.imwrite(r'Sec2\subtracted_image.jpg', sub)
cv2.waitKey(0)

#make border to image
#cv2.copyMakeBorder(image, top, bottom, left, right, optiont of border(cv2.BORDER_CONSTANT,cv2.BORDER_REFLECT,cv2.BORDER_REPLICATE))
'''
The image argument is the input image to which the border will be added.
The top, bottom, left, and right arguments specify the number of pixels to add to each side of the image.
The option of border argument specifies the type of border to add. There are three possible values:
cv2.BORDER_CONSTANT: Adds a constant colored border. This requires an additional argument for the color of the border.
cv2.BORDER_REFLECT: Adds a border that reflects the image across the border.
cv2.BORDER_REPLICATE: Adds a border that replicates the edge pixels of the image.
'''
img4 = cv2.imread(r"Sec2\image.jpg",0 or 1 or -1)
bord = cv2.copyMakeBorder(img4, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
'''
This line adds a 10-pixel-wide constant border to img4 on all sides using the cv2.copyMakeBorder() function. 
The resulting image is stored in the variable bord.
'''
cv2.imshow("bord",bord)
cv2.imwrite(r'Sec2\borded_image.jpg', bord)
cv2.waitKey(0)

# Cropped image
croped_img = img1[100:255, 100:200]
cv2.imshow("croped image",croped_img)
cv2.imwrite(r'Sec2\croped_image.jpg', croped_img)
cv2.waitKey(0)

#Drawing line
#cv2.line(img, start(x,y), end(x,y), BGR, thick)
line = cv2.line(img1, (100, 150), (200, 100), (0, 0, 255), 5)
cv2.imshow("Line", line)
cv2.imwrite(r'Sec2\line.jpg', line)
cv2.waitKey(0)

# Drawing rectangle
#cv2.rectangle(img, start, end, BGR, thick/ fill)
cv2.rectangle(img1, (100, 100), (10, 10), (255, 0, 0), 2)
rectangle = cv2.rectangle(img1, (100, 100), (10, 10), (0, 150, 255), cv2.FILLED)
cv2.imshow("Rectangle", rectangle)
cv2.imwrite(r'Sec2\rectangle.jpg', rectangle)
cv2.waitKey(0)

# Drawing circle
#cv2.circle(img, start, radius, BGR, thick/ fill)
cv2.circle(img1, (200, 100), 10, (0, 0, 255), 5)
circle = cv2.circle(img1, (100, 100), (100), (0, 150, 255), cv2.FILLED)
cv2.imshow("Circle", circle)
cv2.imwrite(r'Sec2\circle.jpg', circle)
cv2.waitKey(0)


# Text
#cv2.putText(img, text, x/y of start, font type, font size, color, font thick)
Text = cv2.putText(img1, "Computer Vision", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
cv2.imshow("Text", Text)
cv2.imwrite(r'Sec2\Text.jpg', Text)
cv2.waitKey(0)


# Translation
M3 = np.float32([[1, 0, 50], [0, 1, 100]])
#trans_img = cv2.warpAffine(img, trans_matrix, (cols(WIDTH OF IMAGE), rows(HIEGHT OF IMAGE)))
trans_img = cv2.warpAffine(img1, M3, (img1.shape[1], img1.shape[0]))
cv2.imshow('Translate', trans_img)
cv2.imwrite(r'Sec2\trans_img.jpg', trans_img)
cv2.waitKey(0)


# Rotation 
#cv2.getRotationMatrix2D(center, angle, scale)
M4 = cv2.getRotationMatrix2D((img1.shape[1] / 2, img1.shape[0] / 2), 45, float(1)) #DIVIDE pixels on 2 to get center of image
rotat_img1 = cv2.warpAffine(img1, M4, (img1.shape[1], img1.shape[0]))
cv2.imshow('Rotation', rotat_img1)
cv2.imwrite(r'Sec2\rotat_img.jpg', rotat_img1)
cv2.waitKey(0)

rotat_img2 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
rotat_img3 = cv2.rotate(img1, cv2.ROTATE_180)
rotat_img4 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow('Rotation', rotat_img1)
cv2.waitKey(0)



# Flipping
# 0 for flipp x axis
# 1 for flipp y axis
# -1 for flipp two axis
flipped = cv2.flip(img1, -1)
cv2.imshow('Flipped', flipped)
cv2.imwrite(r'Sec2\flipped.jpg', flipped)
cv2.waitKey(0)


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


zero = np.zeros(img1.shape[:2],dtype=np.uint8)
red = cv2.merge([zero,zero,r])
gr = cv2.merge([zero,g,zero])
bl = cv2.merge([b,zero,zero])
cv2.imshow('red', red)
cv2.waitKey(0)
cv2.imshow('green', gr)
cv2.waitKey(0)
cv2.imshow('blue', bl)
cv2.waitKey(0)


cv2.destroyAllWindows()



