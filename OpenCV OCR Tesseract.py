import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
# hold control and click pytesseract to see functions available in pytesseract

img = cv2.imread('Kumon_worksheet_Scanned.jpg')
# To resize for kumon size
img = cv2.resize(img, (556,794)) #14.7cm x 21cm # pixel: 555.59055118,793.7007874
_, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
# need to convert BGR to RGB
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# Tesseract Function
print(pytesseract.image_to_string(img2))
print("    ")
    # Format:

# ---------- Detecting Per Digit ------------
hImg,wImg,_ = img.shape
cong = r'--oem 2 --psm 11 outputbase digits'
# oem value 2 --> run cube and tesseract (best accuracy), 3--> default
# psm value 4 --> assume a single column of text of variable sizes, value 6 --> default
# psm value 11 --> Find as much text as possible in no particular order
boxes = pytesseract.image_to_boxes(img2,config=cong) # config=cong
for b in boxes.splitlines():
    #print(b)
    b = b.split(' ') # to split each values based on the space
    # Basically, it transforms into a list
    print(b)
    x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4]) # Convert each string element to int
    cv2.rectangle(img2,(x,hImg-y),(w,hImg-h),(0,0,255),1) # last argument is the thickness
    # Draw the bounding box (ROI)
    cv2.putText(img2,b[0],(x,hImg-y-50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0),0)
    # put the text beside the bounding box

# ---------- Detecting Per Words ------------
# hImg,wImg,_ = img.shape
# boxes = pytesseract.image_to_data(img2)
# print(boxes)
# for x,b in enumerate(boxes.splitlines()):
#     if x!=0:
#         b = b.split()
#         print(b)
# #     b = b.split(' ') # to split each values based on the space
# #     # Basically, it transforms into a list
# #     print(b)
# #     x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4]) # Convert each string element to int
# #     cv2.rectangle(img2,(x,hImg-y),(w,hImg-h),(0,0,255),1) # last argument is the thickness
# #     # Draw the bounding box (ROI)
# #     cv2.putText(img2,b[0],(x,hImg-y-50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0),0)
# #     # put the text beside the bounding box

# Cannot detect MNIST Dataset
cv2.imshow('result',img2)
cv2.waitKey()



