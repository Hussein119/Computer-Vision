"""
Created on 2023

@author: Hussein
"""

import cv2



# Section 1 


#        to read image

# Define variable to save image called img
img1 = cv2.imread(r"Sec1\image.jpg",0 or 1 or -1)

#        to show image

cv2.imshow('name of window',img1) # show as picture
cv2.waitKey(500 or 0)
cv2.destroyAllWindows()
cv2.waitKey(0)
print(img1) # show as matrix

#      to write image to file

cv2.imwrite('Sec1\p2.JPEG',img1)

#      if we want save image under condition 

m1 = cv2.waitKey(0) # m: ASCII code of the pressed key
if m1 == 27:  # 27:  ASCII code for the 'Esc' key,
    cv2.destroyAllWindows()
elif m1 == ord('s'):
   cv2.imwrite('Sec1\p2.PNG',img1)
   cv2.destroyAllWindows()


#In OpenCV, image sampling (the number of pixels in the image) can be changed using the resize function. 
#Note that image size can be specified by the number of rows and columns or by a single Size data type:

img2 = cv2.imread(r"Sec1\p2.PNG",1)
print(img2.shape)
cv2.imshow("Image",img2)
cv2.waitKey(0)

img3 = cv2.resize(img2,(1500,700))
print(img3.shape)

cv2.imshow("Image After re-sizeing",img3)
cv2.waitKey(0)


#           coloring image

img4 = cv2.imread(r"Sec1\image.jpg",0 or 1 or -1)

yuv_image = cv2.cvtColor(img4,cv2.COLOR_BGR2YUV)
gray_img = cv2.cvtColor(img4,cv2.COLOR_BGR2GRAY)

cv2.imshow("gray_img",gray_img)
m2 = cv2.waitKey(0)
cv2.imshow("yuv_image",yuv_image)
m3 = cv2.waitKey(0)
if m2 == 27 or m3 == 27:  # 27:  ASCII code for the 'Esc' key,
    cv2.destroyAllWindows()
elif m2 == ord('s') or m3 == ord('s'):
   cv2.imwrite('Sec1\gray_img.PNG',gray_img)
   cv2.imwrite('Sec1\yuv_image.PNG',yuv_image)
   cv2.destroyAllWindows()











