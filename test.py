from collections import namedtuple
from re import template
from pdf2image import convert_from_path
# Need to install poppler as well
# Docs https://pdf2image.readthedocs.io/en/latest/index.html
import numpy as np
from PIL import Image
import pytesseract
import argparse
import cv2


def pdf_to_image(pdf_path, image_path):
    # main function of converting pdf to image 
    # Set img resolution to 500 dpi
    pages = convert_from_path(pdf_path, 500, None, 1, 1)
    for page in pages:
        page.save(image_path, 'JPEG')

    # Image rotation for better readability for OCR using numpy
    # rotate image 3 times 90 degrees to get an upright position
    input_img = np.array(Image.open(image_path))
    Image.fromarray(np.rot90(input_img, 3)).save(image_path)

def OCR():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image to be OCR'd")
    ap.add_argument("-t", "--template", required=True, help="type of OCR to be performed")
    args = vars(ap.parse_args())

    # Set Boundaries for OCR 
    # For now its Invoice number and the Item List
    OCRLocation = namedtuple('OCRLocation', ['id', 'bbox', 'keywords'])
    OCR_LOCATIONS = [
        OCRLocation('invoice_no', (3276, 1188, 1876, 144), ['Invoice', 'No.']),
        OCRLocation('item_list', (1800, 1808, 656, 1868), ['Brand', 'Description'])
                     ]

    print("[INFO] loading images")
    image = cv2.imread(args["image"])
    template=cv2.imread(args["template"])

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


                
if __name__ == "__main__":
    IMG_PATH = './test.jpg'
    PDF_PATH = 'assets/test.pdf'
    pdf_to_image(PDF_PATH, IMG_PATH)
    OCR()
