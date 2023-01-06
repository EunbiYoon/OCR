import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR/tesseract.exe"
img=cv2.imread('plate.png')

#1차 개선 사이즈 조절
img=cv2.resize(img, None, fx=0.5, fy=0.5)
#2차 gray로 개선 색깔 전환
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#3차 배경과 텍스트를 구분해 주자 threshold 사용
adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11)

#image to string
config='--psm 4'
text=pytesseract.image_to_string(adaptive_threshold,config=config)
print(text)

# image show
cv2.imshow('Img',img)
cv2.imshow('gray',gray)
cv2.imshow('adaptive_threshold',adaptive_threshold)

cv2.waitKey(0)
