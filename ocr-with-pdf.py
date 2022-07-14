from collections import namedtuple
from email.mime import image
from re import template
from pdf2image import convert_from_path
# Need to install poppler as well
# Docs https://pdf2image.readthedocs.io/en/latest/index.html
import numpy as np
from PIL import Image
import pytesseract
import cv2
import imutils
import glob


def pdf_to_image(pdf_path, image_counter):
    # main function of converting pdf to image 
    # Set img resolution to 500 dpi
    pages = convert_from_path(pdf_path, 350)
    for page in pages:
        
        if image_counter%2 == 0:
            filename = "page_" + str(image_counter) + ".jpg"
            image_path = "./" + filename
            page.save(image_path, 'JPEG')
        image_counter += 1
        # Image rotation for better readability for OCR using numpy
        # rotate image 3 times 90 degrees to get an upright position
        #input_img = np.array(Image.open(image_path))
        #Image.fromarray(np.rot90(input_img, 3)).save(image_path)
    return image_counter

def OCR(fileLimit):
   # Set Boundaries for OCR 
    # For now its Invoice number and the Item List
    OCRLocation = namedtuple('OCRLocation', ['id', 'bbox', 'keywords'])
    OCR_LOCATIONS = [
        OCRLocation('invoice_no', (3328, 1155, 1809, 99), ['Billing']),
        OCRLocation('item_list', (1881, 1744, 656, 566), ['Brand', 'Description']),
        OCRLocation('qty',(2987,1821,1821,566), ['Qty'])
                     ]
    for image_index in range(1, fileLimit):
        if image_index%2 == 0:
            print("[INFO] loading images")
            image_file = "./page_" + str(image_index) + ".jpg"
            image = cv2.imread(image_file)
    

            print("[INFO] OCR in progress")
            res = []
            for loc in OCR_LOCATIONS:
                (x, y, w, h) = loc.bbox
                roi = image[y:y+h, x:x+w]
                rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                text = pytesseract.image_to_string(rgb)
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
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                for (i, line ) in enumerate(text.split("\n")):
                    if i != 0:
                        y = y+ (i*70) +40
                    cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 5)
            cv2.imshow("Image", image)
            cv2.waitKey(0)


def pre_process_image(img, save_in_file, morph_size=(8, 8)):

    # get rid of the color
    pre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu threshold
    pre = cv2.threshold(pre, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # dilate the text to make it solid spot
    cpy = pre.copy()
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size)
    cpy = cv2.dilate(~cpy, struct, anchor=(-1, -1), iterations=1)
    pre = ~cpy

    if save_in_file is not None:
        cv2.imwrite(save_in_file, pre)
    return pre


if __name__ == "__main__":
    image_counter = 1
    PDF_PATH = 'assests/PDF/1.pdf'
    fileLimit = pdf_to_image(PDF_PATH, image_counter)
    OCR(fileLimit)
