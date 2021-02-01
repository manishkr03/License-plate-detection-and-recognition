#pytessearct ocr to extract text from image
import cv2
import pytesseract
import imutils

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Resize, grayscale, Gaussian blur, Otsu's threshold
image = cv2.imread('imgPlate.png')

y=5
x=5
h=image.shape[1]-18
w=image.shape[0]
image = image[x:w, y:h]

image = imutils.resize(image, width=400)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Perform morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=3)


# Invert, Blur, and perform text extraction
invert = 255 - cv2.GaussianBlur(close, (3,3), 0)
data = pytesseract.image_to_string(invert, lang='eng',config='--psm 6')
print(data)

cv2.imshow('thresh', thresh)
cv2.imshow('opening', opening)
cv2.imshow('close', close)
cv2.imshow('invert', invert)
cv2.waitKey(0) & 0xFF == ord('q')
cv2.destroyAllWindows()



#bounding box on each text and apply ocr

import cv2
import pytesseract
import imutils

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Resize, grayscale, Gaussian blur, Otsu's threshold
image = cv2.imread('imgPlate.png')

y=5
#x=0
x=5
h=image.shape[1]-18
w=image.shape[0]
image = image[x:w, y:h]

image = imutils.resize(image, width=400)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Perform morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=3)

# Invert, Blur, and perform text extraction
invert = 255 - cv2.GaussianBlur(close, (3,3), 0)

ROI_number = 0
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

count=0
for c in cnts:
    area = cv2.contourArea(c)
    print(count, ":", area)
    count=count+1
    if area>=351:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        ROI = image[y:y+h, x:x+w]
        cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
        ROI_number += 1
    

print(len(cnts))
data = pytesseract.image_to_string(invert, lang='eng',config='--psm 6')
print(data)

cv2.imshow("image", image)
cv2.imshow('thresh', thresh)
cv2.imshow('opening', opening)
cv2.imshow('close', close)
cv2.imshow('invert', invert)
cv2.imwrite('invert.png', invert)

cv2.waitKey(0) & 0xFF == ord('q')
cv2.destroyAllWindows()


###########################
"""
import cv2
import pytesseract
import imutils

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Resize, grayscale, Gaussian blur, Otsu's threshold
image = cv2.imread('imgPlate.png')

y=5
x=0
h=image.shape[1]-18
w=image.shape[0]
image = image[x:w, y:h]

image = imutils.resize(image, width=400)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Perform morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=3)


# Invert, Blur, and perform text extraction
invert = 255 - cv2.GaussianBlur(close, (3,3), 0)
data = pytesseract.image_to_string(invert, lang='eng',config='--psm 6')
print(data)

cv2.imshow('thresh', thresh)
cv2.imshow('opening', opening)
cv2.imshow('close', close)
cv2.imshow('invert', invert)
cv2.waitKey(0) & 0xFF == ord('q')
cv2.destroyAllWindows()
"""