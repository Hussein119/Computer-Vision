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


# # # Merge
# # image_merge = cv2.merge([b, g, r])
# # cv2.imshow("RGB_Image", image_merge)
# # square=np.zeros((400,400),dtype="uint8")
# # img_rec=cv2.rectangle(square,(50,50),(350,350),255,-1)
# # cv2.imshow("circle",img_rec)


# # circle=np.zeros((400,400),dtype="uint8")
# # img_cir=cv2.circle(circle,(200,200),200,255,-1)

# # # And operation 
# # and_img = cv2.bitwise_and(img_rec, img_cir, mask = None)
# # cv2.imshow("AND", and_img)

# # #Or operation
# # or_img = cv2.bitwise_or(img_rec, img_cir)
# # cv2.imshow("OR", or_img)

# # #Xor operation
# # xor_img = cv2.bitwise_xor(img_rec, img_cir)
# # cv2.imshow("XOR", xor_img)

# # #Not operation
# # not_img = cv2.bitwise_not(img_cir)
# # cv2.imshow("NOT", not_img)

# # Image masking
# # mask = np.zeros(img.shape[:2], dtype = np.uint8) #8-bit unsigned integer (0 to 255)
# # center = (img.shape[1] // 2, img.shape[0] // 2) 

# # cv2.circle(mask, center, 200, 255, -1)
# # cv2.imshow("Mask", mask)

# # img_msk = cv2.bitwise_and(img, img, mask = mask)
# # cv2.imshow("Masked image", img_msk)

# # mouse event function
# def click_event(event, x, y, flags, param):
    
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(x, ', ', y)
#         strxy = str(x) + ', ' + str(y)
#         cv2.putText(img, strxy, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
#         cv2.imshow("Michky", img)
        
#     if event == cv2.EVENT_RBUTTONDOWN:
#         b = img[y, x, 0]
#         g = img[y, x, 1]
#         r = img[y, x, 2]
#         strbgr = str(b) + ', ' + str(g) + ', ' + str(r)
#         cv2.putText(img, strbgr, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
#         cv2.imshow("Michky", img)    
        
# cv2.imshow("Michky", img)
# cv2.setMouseCallback("Michky", click_event)
        
# # Matplotlib
# img_mpl = plt.imread(r"D:\computer vision\dogs.jpg")     
# #plt.imshow("asd",img_mpl)
# cv2.imshow("asd",img_mpl)
# histr = cv2.calcHist([img],[2],None,[256],[0,256]) 
# plt.plot(histr)
# #plt.imshow()
# cv2.waitKey(0)