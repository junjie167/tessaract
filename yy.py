import cv2
import numpy as np
import pytesseract
import math
from imutils import contours

KEYWORD_HEADER = ['Brand Description', 'Qty', 'Unit Price', 'GST', 'Line Price', 'Invoice No.', 'Billing']

# Load image, grayscale, Gaussian blur, Otsu's threshold
pre = cv2.imread('test_product 2.jpg')
image = cv2.imread('test_product 2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Detect horizontal lines
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
horizontal_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
# contours,h = cv2.findContours(horizontal_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# rows = [None]*len(contours)
# print(len(contours))
# for i, c in enumerate(contours):
#     rows[i] = cv2.boundingRect(cv2.approxPolyDP(c, 3, True))
#     cv2.drawContours(image, [c], -1, (36,255,12), 5)  
# rows = sorted(rows, key=lambda b:b[1], reverse=False)


# cnts = cv2.findContours(horizontal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])
# for c in cnts:
#     cv2.drawContours(image, [c], -1, (36,255,12), 5)


# Detect vertical lines
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,50))
vertical_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
# contours,h = cv2.findContours(vertical_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# cols = [None]*len(contours)
# for i, c in enumerate(contours):
#     cols[i] = cv2.boundingRect(cv2.approxPolyDP(c, 3, True))
#     print(cols[i])
#     cv2.drawContours(image, [c], -1, (36,255,12), 5)
    
# cols = sorted(cols, key=lambda b:b[0], reverse=False)
# cnts = cv2.findContours(vertical_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])
# for c in cnts:
#     cv2.drawContours(image, [c], -1, (36,255,12), 5)
# cv2.imwrite("test_product2.jpg", image)


# Combine masks and remove lines
table_mask = cv2.bitwise_or(horizontal_mask, vertical_mask)

image[np.where(table_mask==255)] = [255,255,255]

cv2.imshow('image', image)
#cv2.imwrite("test_product.jpg", image)
cv2.waitKey()


cont, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cnts = cont[0] if len(cont) == 2 else cont[1]

# cnts = sorted(cont, key=lambda x: cv2.boundingRect(x)[0])
# cnts = sorted(cont, key=lambda x: cv2.boundingRect(x)[0])
cnts, _ = contours.sort_contours(cont, method="left-to-right")
cnts, _ = contours.sort_contours(cnts, method="top-to-bottom")
# tableCNT = max(np.array(cnts), key=cv2.contourArea)
# max_area = cv2.contourArea(np.array(tableCNT))
# # def get_boxes(num, method="left-to-right"):
# #      # initialize the reverse flag and sort index
# #     reverse = False
# #     i = 0

# #     # handle if we need to sort in reverse
# #     if method == "right-to-left" or method == "bottom-to-top":
# #         reverse = True

# #     # handle if we are sorting against the y-coordinate rather than
# #     # the x-coordinate of the bounding box
# #     if method == "top-to-bottom" or method == "bottom-to-top":
# #         i = 1

# #     # construct the list of bounding boxes and sort them from top to
# #     # bottom
# boundingBoxes = [cv2.boundingRect(c) for c in cont]
# (contours, boundingBoxes) = zip(*sorted(zip(cont, boundingBoxes),
# key=lambda x:x[1][1]))

boxes = []
for contour in cnts:
    # b = cv2.contourArea(np.array(contour))
    # if not math.isclose(max_area - b, 0.0):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.drawContours(image, [contour], -1, (36,255,12), 5)
    #image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        boxes.append([x,y,w,h])



start_point = None
item_dict = {}
for xb in boxes:

    
    x, y ,w, h = xb
    if w >= 100 and h >= 50 and h < 300:
        rw = image[y:y+h , x:x+w]
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        # border = cv2.copyMakeBorder(rw,2,2,2,2, cv2.BORDER_CONSTANT,value=[255,255])
        # resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        # dilation = cv2.dilate(resizing, kernel,iterations=1)
        # erosion = cv2.erode(dilation, kernel,iterations=2)
        #rw = cv2.cvtColor(rw, cv2.COLOR_BGR2GRAY)
        print(xb)
        text = pytesseract.image_to_string(rw, config="--psm 6")
        for keyword in KEYWORD_HEADER:
            if keyword in text:
                start_point = keyword
                value_list = []
                break

        if start_point != None and not start_point in text:
            value_list.append(text)
            item_dict[start_point] = value_list
        
        print([text])
        cv2.imshow('image', rw)
        #cv2.imwrite("test_product.jpg", image)
        cv2.waitKey()

print(item_dict)
# rows=[]
# columns=[]
# heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
# mean = np.mean(heights)
# print(mean)
# columns.append(boxes[0])
# previous=boxes[0]
# for i in range(1,len(boxes)):
#     if(boxes[i][1]<=previous[1]+mean/2):
#         columns.append(boxes[i])
#         previous=boxes[i]
#         if(i==len(boxes)-1):
#             rows.append(columns)
#     else:
#         rows.append(columns)
#         columns=[]
#         previous = boxes[i]
#         columns.append(boxes[i])
# print("Rows")
# for row in rows:
#     print(row)

# total_cells=0
# for i in range(len(row)):
#     if len(row[i]) > total_cells:
#         total_cells = len(row[i])
# print(total_cells)

# center = [int(rows[i][j][0]+rows[i][j][2]/2) for j in range(len(rows[i])) if rows[0]]
# print(center)
# center=np.array(center)
# center.sort()
# print(center)

# boxes_list = []
# for i in range(len(rows)):
#     l=[]
#     for k in range(total_cells):
#         l.append([])
#     for j in range(len(rows[i])):
#         diff = abs(center-(rows[i][j][0]+rows[i][j][2]/4))
#         minimum = min(diff)
#         indexing = list(diff).index(minimum)
#         l[indexing].append(rows[i][j])
#     boxes_list.append(l)
# for box in boxes_list:
#     print(box)

# dataframe_final=[]
# for i in range(len(boxes_list)):
#     for j in range(len(boxes_list[i])):
#         s=''
#         if(len(boxes_list[i][j])==0):
#             dataframe_final.append(' ')
#         else:
#             for k in range(len(boxes_list[i][j])):
#                 y,x,w,h = boxes_list[i][j][k][0],boxes_list[i][j][k][1], boxes_list[i][j][k][2],boxes_list[i][j][k][3]
#                 roi = image[x:x+h, y:y+w]
#                 kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
#                 border = cv2.copyMakeBorder(roi,2,2,2,2, cv2.BORDER_CONSTANT,value=[255,255])
#                 resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#                 dilation = cv2.dilate(resizing, kernel,iterations=1)
#                 erosion = cv2.erode(dilation, kernel,iterations=2)
#                 out = pytesseract.image_to_string(roi, config='--psm 4')
#                 print(out)
#                 if(len(out)==0):
#                     out = pytesseract.image_to_string(roi, config='--psm 4')
#                     print(out)
#                 s = s +" "+ out
#             dataframe_final.append(s)
# print(dataframe_final)

# cnts = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])
# # tableCnt = max(cnts, key=cv2.contourArea)
# # (x, y, w, h) = cv2.boundingRect(tableCnt)
# # table = image[y:y + h, x:x + w]
# list_coordinate = []
# for c in cnts:
#     x,y,w,h = cv2.boundingRect(c)
#     print(x,y,w,h)
#     cv2.drawContours(image, [c], -1, (36,255,12), 5)
#     list_coordinate.append([(x,y), (x+w, y+h)])
# cv2.imshow('image', image)
# #cv2.imwrite("test_product.jpg", image)
# cv2.waitKey()
# c = list_coordinate[1]
# productTable = image[c[0][1]:c[1][1], c[0][0]:c[1][0]]
# cv2.imshow('image', productTable)
# cv2.imwrite("test_product2.jpg", image)
# cv2.waitKey()



# img = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
# contours,h = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours_poly = [None]*len(contours)
# boundRect = [None]*len(contours)

# for i, c in enumerate(contours):
#     contours_poly[i] = cv2.approxPolyDP(c, 3, True)
#     boundRect[i] = cv2.boundingRect(contours_poly[i])
# cv2.imshow('image', image)
# #cv2.imwrite("test_product.jpg", image)
# cv2.waitKey()
# ext_position = []
# offest = 10
# boundingBoxes = sorted(boundRect, key=lambda b:b[0], reverse=False)
# print(boundingBoxes)


# for rect in boundingBoxes:
    
#     image = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
#     text = pytesseract.image_to_string(image)
#     cv2.imshow('image', image)
#     #cv2.imwrite("test_product.jpg", image)
#     cv2.waitKey()
#     for i,row in enumerate(rows):
#         if i < len(rows):
#             if rect[1] >row[1] and rect[1] <rows[i+1][1]:
#                 r = i 
#                 break 
#     for i,col in enumerate(cols):
#         if i < len(cols):
#             if rect[0] >col[0] and rect[0] <cols[i+1][0]:
#                 c = i 
#                 break

# cv2.imshow('thresh', thresh)
cv2.imshow('horizontal_mask', horizontal_mask)
cv2.imshow('vertical_mask', vertical_mask)
cv2.imshow('table_mask', table_mask)
cv2.imshow('image', image)
#cv2.imwrite("test_product.jpg", image)
cv2.waitKey()
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Function to sanitize text
# def sanitize_text(image_path):
#     # Store the extracted text from the image in a string.
#     # Then, split the string into an array using the '\n' character as a delimiter
#     raw_text_values = pytesseract.image_to_string(image_path, config="--psm 4").split('\n')
#     # Remove any spaces or empty values
#     for i in range(len(raw_text_values) - 1, -1, -1):
#         if raw_text_values[i] == '' or raw_text_values[i] == ' ':
#             del raw_text_values[i]
#     return raw_text_values


import pytesseract

text = pytesseract.image_to_string(image)

print([text])
print('\n\n' in text)

# cont, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# def get_boxes(num, method="left-to-right"):
#      # initialize the reverse flag and sort index
#     reverse = False
#     i = 0

#     # handle if we need to sort in reverse
#     if method == "right-to-left" or method == "bottom-to-top":
#         reverse = True

#     # handle if we are sorting against the y-coordinate rather than
#     # the x-coordinate of the bounding box
#     if method == "top-to-bottom" or method == "bottom-to-top":
#         i = 1

#     # construct the list of bounding boxes and sort them from top to
#     # bottom
#     boundingBoxes = [cv2.boundingRect(c) for c in num]
#     (num, boundingBoxes) = zip(*sorted(zip(num, boundingBoxes),
#         key=lambda b:b[1][i], reverse=reverse))

#     # return the list of sorted contours and bounding boxes
#     return (num, boundingBoxes)
# cont, boxes = get_boxes(cont, method="top-to-bottom")

# final_box = []
# for c in cont:
#     s1, s2, s3, s4 = cv2.boundingRect(c)
    
#     rectangle_img = cv2.rectangle(pre,(s1,s2),(s1+s3,s2+s4),(0,0,255),5)
#     final_box.append([s1,s2,s3,s4])
# graph = plt.imshow(rectangle_img,cmap='gray')
# plt.show()

# dim = [final_box[i][3] for i in range(len(final_box))]
# avg = np.mean(dim)
# hor=[]
# ver=[]
# for i in range(len(final_box)):    
#     if(i==0):
#         ver.append(final_box[i])
#         last=final_box[i]    
#     else:
#         if(final_box[i][1]<=last[1]+avg/2):
#             ver.append(final_box[i])
#             last=final_box[i]            
#             if(i==len(final_box)-1):
#                 hor.append(ver)        
#         else:
#             hor.append(ver)
#             ver=[]
#             last = final_box[i]
#             ver.append(final_box[i])
# total = 0
# for i in range(len(hor)):
#     total = len(hor[i])
#     if total > total:
#         total = total
# mid = [int(hor[i][j][0]+hor[i][j][2]/2) for j in range(len(hor[i])) if hor[0]]
# mid=np.array(mid)
# mid.sort()

# order = []
# for i in range(len(hor)):
#     arrange=[]
#     for k in range(total):
#         arrange.append([])
#     for j in range(len(hor[i])):
#         sub = abs(mid-(hor[i][j][0]+hor[i][j][2]/4))
#         lowest = min(sub)
#         idx = list(sub).index(lowest)
#         arrange[idx].append(hor[i][j])
#     order.append(arrange)

# try:
#     from PIL import Image
# except ImportError:
#     import Image
# import pytesseract

# extract=[]
# for i in range(len(order)):
#     for j in range(len(order[i])):
#         inside=''
#         if(len(order[i][j])==0):
#             extract.append(' ')
#         else:
#             for k in range(len(order[i][j])):
#                 side1,side2,width,height = order[i][j][k][0],order[i][j][k][1], order[i][j][k][2],order[i][j][k][3]
#                 final_extract = pre[side2:side2+height, side1:side1+width]
#                 final_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#                 get_border = cv2.copyMakeBorder(final_extract,2,2,2,2, cv2.BORDER_CONSTANT,value=[255,255])
#                 resize = cv2.resize(get_border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#                 dil = cv2.dilate(resize, final_kernel,iterations=1)
#                 ero = cv2.erode(dil, final_kernel,iterations=2)
#                 ocr = pytesseract.image_to_string(ero, config="--psm 6")
#                 inside = inside +" "+ ocr
#                 extract.append(inside)
                    
                    

# print(extract)

