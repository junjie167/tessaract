import cv2
import numpy as np

def rectify(h):
        h = h.reshape((4,2))
        hnew = np.zeros((4,2),dtype = np.float32)

        add = h.sum(1)
        hnew[0] = h[np.argmin(add)]
        hnew[2] = h[np.argmax(add)]

        diff = np.diff(h,axis = 1)
        hnew[1] = h[np.argmin(diff)]
        hnew[3] = h[np.argmax(diff)]

        return hnew

img = cv2.imread('test_product 3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# GaussianBlur(src, ksize, sigma1[, dst[, sigma2[, borderType]]]) -> dst
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) -> dst
thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Biggest blob is the square we are looking for. Results in 4x1x2 array.
# First row is the TOP-RIGHT corner. Second row is the TOP-LEFT corner. Third row is the BOTTOM-LEFT corner. Finally, fourth one is the BOTTOM-RIGHT corner. 
# The problem is that, there is no guarantee that for next image, the corners found out will be in this same order.
# Change to uniform order with rectify. [TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT]
biggest = None
max_area = 0
for i in contours:
    area = cv2.contourArea(i)
    if area > 100:
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.02 * peri, True)
        if area > max_area and len(approx) == 4:
            biggest = approx
            max_area = area

print(approx.shape)            
approx = rectify(biggest)

h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)



cv2.imshow('image', approx)
cv2.waitKey(0)