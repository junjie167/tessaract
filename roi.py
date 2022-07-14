import cv2
from PIL import Image
import pytesseract
#4350 730
def mark_region(imagE_path):
    
    im = cv2.imread(imagE_path)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur,0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])

    line_items_coordinates = []
    he = 0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        #print(w, h)
        print(x)
        # if w >= 3000 and h<= 570:
        image = cv2.rectangle(im, (x,y), (x+w, y+h), (255,0,255), 5)
        he =h
        line_items_coordinates.append([(x,y), (x+w, y+h)])

    return image, line_items_coordinates, he

i, li, he = mark_region("page_2.jpg")

print(li)

cv2.imshow("Image", i)
cv2.waitKey(0)

c = li[0]

im = i[c[0][1]:c[1][1], c[0][0]:c[1][0]]
print(he)

#i = im[40:40+668, 1472:1472+610]
#i = im[18:18+668, 2562:2562+256]
#i = img[1227:1227+297, 1331:1331+424]
#i = img[1221:1221+292, 2140:2140+130]
#i = img[1216:1216+302, 2387:2387+715]
i = im[6:6+472, 2596:2596+258]
cv2.imwrite("test.jpg", im)

# convert the image to black and white for better OCR
ret,thresh1 = cv2.threshold(i,120,255,cv2.THRESH_BINARY)

# pytesseract image to string to get results
text = str(pytesseract.image_to_string(thresh1, config="--psm 6"))
print(text)



