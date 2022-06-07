from pdf2image import convert_from_path
# Need to install poppler as well
# Docs https://pdf2image.readthedocs.io/en/latest/index.html
import numpy as np
from PIL import Image
import pytesseract
import argparse
import imultils


def pdf_to_image(pdf_path, image_path):
    # main function of converting pdf to image 
    # Set img resolution to 500 dpi
    pages = convert_from_path(pdf_path, 500)
    for page in pages:
        page.save(image_path, 'JPEG')

    # Image rotation for better readability for OCR using numpy
    # rotate image 3 times 90 degrees to get an upright position
    input_img = np.array(Image.open(image_path))
    Image.fromarray(np.rot90(input_img, 3)).save(image_path)
    
if __name__ == "__main__":
    IMG_PATH = './test.jpg'
    PDF_PATH = 'assets/test.pdf'
    pdf_to_image(PDF_PATH, IMG_PATH)
