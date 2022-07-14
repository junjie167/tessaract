import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

read_image= cv2.imread("test_align.jpg",0)

convert_bin,grey_scale = cv2.threshold(read_image,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
grey_scale = 255-grey_scale
grey_graph = plt.imshow(grey_scale,cmap='gray')
plt.show()

length = np.array(read_image).shape[1]//100
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))

horizontal_detect = cv2.erode(grey_scale, horizontal_kernel, iterations=3)
hor_line = cv2.dilate(horizontal_detect, horizontal_kernel, iterations=3)
plotting = plt.imshow(horizontal_detect,cmap='gray')
plt.show()

vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
vertical_detect = cv2.erode(grey_scale, vertical_kernel, iterations=3)
ver_lines = cv2.dilate(vertical_detect, vertical_kernel, iterations=3)
show = plt.imshow(vertical_detect,cmap='gray')
plt.show()

final = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
combine = cv2.addWeighted(ver_lines, 0.5, hor_line, 0.5, 0.0)
combine = cv2.erode(~combine, final, iterations=2)
thresh, combine = cv2.threshold(combine,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
convert_xor = cv2.bitwise_xor(read_image,combine)
inverse = cv2.bitwise_not(convert_xor)
output= plt.imshow(inverse,cmap='gray')
plt.show()

cont, _ = cv2.findContours(combine, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
def get_boxes(num, method="left-to-right"):
     # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in num]
    (num, boundingBoxes) = zip(*sorted(zip(num, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (num, boundingBoxes)
cont, boxes = get_boxes(cont, method="top-to-bottom")

final_box = []
for c in cont:
    s1, s2, s3, s4 = cv2.boundingRect(c)
    print(s3, s4)
    if (s3<=1809 and s4<=99):
        rectangle_img = cv2.rectangle(read_image,(s1,s2),(s1+s3,s2+s4),(0,0,255),5)
        final_box.append([s1,s2,s3,s4])
graph = plt.imshow(rectangle_img,cmap='gray')
plt.show()

dim = [final_box[i][3] for i in range(len(final_box))]
avg = np.mean(dim)
hor=[]
ver=[]
for i in range(len(final_box)):    
    if(i==0):
        ver.append(final_box[i])
        last=final_box[i]    
    else:
        if(final_box[i][1]<=last[1]+avg/2):
            ver.append(final_box[i])
            last=final_box[i]            
            if(i==len(final_box)-1):
                hor.append(ver)        
        else:
            hor.append(ver)
            ver=[]
            last = final_box[i]
            ver.append(final_box[i])
total = 0
for i in range(len(hor)):
    total = len(hor[i])
    if total > total:
        total = total
mid = [int(hor[i][j][0]+hor[i][j][2]/2) for j in range(len(hor[i])) if hor[0]]
mid=np.array(mid)
mid.sort()

order = []
for i in range(len(hor)):
    arrange=[]
    for k in range(total):
        arrange.append([])
    for j in range(len(hor[i])):
        sub = abs(mid-(hor[i][j][0]+hor[i][j][2]/4))
        lowest = min(sub)
        idx = list(sub).index(lowest)
        arrange[idx].append(hor[i][j])
    order.append(arrange)

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

extract=[]
for i in range(len(order)):
    for j in range(len(order[i])):
        inside=''
        if(len(order[i][j])==0):
            extract.append(' ')
        else:
            for k in range(len(order[i][j])):
                side1,side2,width,height = order[i][j][k][0],order[i][j][k][1], order[i][j][k][2],order[i][j][k][3]
                final_extract = read_image[side2:side2+height, side1:side1+width]
                final_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                get_border = cv2.copyMakeBorder(final_extract,2,2,2,2, cv2.BORDER_CONSTANT,value=[255,255])
                resize = cv2.resize(get_border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                dil = cv2.dilate(resize, final_kernel,iterations=1)
                ero = cv2.erode(dil, final_kernel,iterations=2)
                ocr = pytesseract.image_to_string(ero, config="--psm 6")
                inside = inside +" "+ ocr
                extract.append(inside)
                    
                    

print(extract)