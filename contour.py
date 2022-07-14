import cv2
import numpy as np
import matplotlib.pyplot as plt

# 4300 , h<=720
#1750, 1850, h<=650
img1 = cv2.imread("page_1.jpg")
ret, thresh_value = cv2.threshold(
    img1, 180, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((3,3),np.uint8)
dilated_value = cv2.dilate(thresh_value,kernel,iterations = 1)
img = cv2.cvtColor(dilated_value, cv2.COLOR_BGR2GRAY)
contours, hierarchy = cv2.findContours(
    img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
line_items_coordinates = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    # bounding the images
    print(x, w, h)
    if x >= 400 and x<= 500 and w >= 700 and y >= 70 and y <= 110:
        img1 = cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 1)
        line_items_coordinates.append([(x,y), (x+w, y+h)])

c = line_items_coordinates[0]
im = img1[c[0][1]:c[1][1], c[0][0]:c[1][0]]

#cv2.imshow("Image", im)
#cv2.waitKey(0)
plt.imshow(im)
plt.show()