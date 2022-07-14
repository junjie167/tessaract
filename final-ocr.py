from collections import namedtuple
from re import template
from pdf2image import convert_from_path
# Need to install poppler as well
# Docs https://pdf2image.readthedocs.io/en/latest/index.html
import numpy as np
from PIL import Image
import pytesseract
import cv2

#OCRLocation('item_list', (1881, 1744, 656, 566), ['Brand', 'Description']),
#OCRLocation('qty',(2987,1821,1821,566), ['Qty'])
#OCRLocation('Brand_Description', (1328, 1232, 416, 0), ['Brand', 'Description'])

def pdf_to_image(pdf_path, image_counter):
    # main function of converting pdf to image 
    # Set img resolution to 500 dpi
    pages = convert_from_path(pdf_path, 500)
    for page in pages:
        
        if image_counter%2 == 0:
            filename = "page_" + str(image_counter) + ".jpg"
            image_path = "./" + filename
            page.save(image_path, 'JPEG')
            # Image rotation for better readability for OCR using numpy
            # rotate image 3 times 90 degrees to get an upright position
            #input_img = np.array(Image.open(image_path))
            #Image.fromarray(np.rot90(input_img, 3)).save(image_path)
        image_counter += 1
        
    return image_counter

def pre_process_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur,0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    return dilate

def mark_region(image, locationID, height):
    ret, thresh_value = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3),np.uint8)
    dilated_value = cv2.dilate(thresh_value,kernel,iterations = 1)
    img = cv2.cvtColor(dilated_value, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    line_items_coordinates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # bounding the images
        #Mark Product Table
        if w >= 3000 and h<= 510 and locationID is None:
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)
            height = h
            line_items_coordinates.append([(x,y), (x+w, y+h)])
            break
        elif x >= 1035 and x<= 1085 and locationID == "Brand_Description":
            h = height - 200
            image = cv2.rectangle(image, (x, 20), (x + w, 20 + h), (0, 0, 255), 5)
            line_items_coordinates.append([(x,20), (x+w, 20+h)])
            break
        elif x >= 1800 and x <= 1870 and w >= 165 and w <= 180 and locationID == "Qty":
            h = height - 200
            image = cv2.rectangle(image, (x,13), (x + w, 13 + h), (0, 0, 255), 5)
            line_items_coordinates.append([(x,13), (x+w, 13+h)])
            break
        elif x >= 2090 and x <= 2180 and locationID == "Unit_Price":
            h = height - 200
            image = cv2.rectangle(image, (x, 20), (x + w, 20 + h), (0, 0, 255), 5)
            line_items_coordinates.append([(x,20), (x+w, 20+h)])
            break

    return image, line_items_coordinates, height

def OCR_Digit(roi):
    grayImage = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    (h, w) = grayImage.shape[:2]
    grayImage = cv2.resize(grayImage, (w*2, h*2))
    grayImage = grayImage[30:(h*2), w+50:(w*2)]
    thr1 = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cv2.imshow("Image", grayImage)
    cv2.waitKey(0)
    text = pytesseract.image_to_string(thr1, config="--psm 6 digits")
    print(text)
    return text 

def OCR(total_num_invoices):
   # Set Boundaries for OCR 
    # For now its Invoice number and the Item List
    OCRLocation = namedtuple('OCRLocation', ['id', 'bbox', 'keywords'])
    OCR_LOCATIONS = [
        OCRLocation('invoice_no', (2316, 810, 1264, 65), ['Billing']),
        OCRLocation('Brand_Description', (1035, 35, 413, 0), ['Brand', 'Description']),
        OCRLocation('Qty', (1824,13,163,0), ['Qty']),
        OCRLocation('Unit_Price', (2160, 13, 193, 0), ['Unit Price'])
                     ]
    for image_index in range(1, total_num_invoices):
        if image_index%2 == 0:
            print("[INFO] loading images")
            image_file = "./page_" + str(image_index) + ".jpg"
            image = cv2.imread(image_file)

            print("Marking Region for Product Table")

            img_with_product_region, coordinate, height = mark_region(image, None, None)
            #OCR_LOCATIONS.append(('item_list', (1881, 1744, 656, height-154), ['Brand', 'Description']))
            print(height)
            print("[INFO] OCR in progress")
            res = []
            for loc in OCR_LOCATIONS:
                (x, y, w, h) = loc.bbox
                if loc.id == "invoice_no":
                    roi = image[y:y+h, x:x+w]
                else:
                    c = coordinate[0]
                    productTable = img_with_product_region[c[0][1]:c[1][1], c[0][0]:c[1][0]]
                    product_attribute_region, attribute_coordinate, height = mark_region(productTable, loc.id, height)
                    ac = attribute_coordinate[0]
                    roi = product_attribute_region[ac[0][1]:ac[1][1], ac[0][0]:ac[1][0]]
            
                if loc.id == "Brand_Description" or loc.id == "invoice_no":
                    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    text = pytesseract.image_to_string(rgb)
                else:
                    text = OCR_Digit(roi)

                for line in text.split("\n"):
                    if len(line) == 0:
                        continue
                    count = sum([line.count(x) for x in loc.keywords])
                    if count == 0:
                        res.append((loc,line))
            final_res = {}
            for(loc, line) in res:
                r = final_res.get(loc.id,None)
                if r is None:
                    final_res[loc.id] =(line, loc._asdict())
                else:
                    (exist, loc) = r
                    text = "{}\n{}".format(exist, line)
                    final_res[loc["id"]] = (text, loc)
    
            # OCR Visualization
            for (locID, result) in final_res.items():
                (text, loc) = result
                print(loc["id"])
                print("=" * len(loc["id"]))
                print("{}\n\n".format(text))

                # Bounding box visualisation
                (x, y, w, h) = loc["bbox"]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
                for (i, line ) in enumerate(text.split("\n")):
                    if i != 0:
                        y = y+ (i*70) +40
                    cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 5)
            cv2.imshow("Image", image)
            cv2.waitKey(0)


if __name__ == "__main__":
    image_counter = 1
    PDF_PATH = 'assests/PDF/1.pdf'
    total_num_invoices = pdf_to_image(PDF_PATH, image_counter)
    # OCR(total_num_invoices)