import os
import cv2 as cv
import numpy as np


faceCascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
mouthCascade = cv.CascadeClassifier('haarcascade_mcs_mouth.xml')
noseCascade = cv.CascadeClassifier('haarcascade_mcs_nose.xml')

no_face_found = "No face detected..."
weared_mask = "Mask detected"
weared_mask_incorrectly = "Make sure your mask is covering your nose and mouth."
not_weared_mask = "No mask detected"

cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10, 150)

while True:
    success, img = cap.read()
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 2)
    if(len(faces) == 0):
        cv.putText(img, no_face_found, (30,30), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        mouths = mouthCascade.detectMultiScale(imgGray, 1.7, 11)
        noses = noseCascade.detectMultiScale(imgGray, 1.1, 2)
        if(len(mouths) == 0 and len(noses) == 0):
            cv.putText(img, weared_mask, (30,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv.LINE_AA)
        elif(len(mouths) == 0 or len(noses) == 0):
            cv.putText(img, weared_mask_incorrectly, (30,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv.LINE_AA)
        else:
            cv.putText(img, not_weared_mask, (30,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv.LINE_AA)
        for (x,y,w,h) in mouths:
            y = int(y - 0.15*h)
            cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            break
        for (x, y, w, h) in noses:
            cv.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
            break

    cv.imshow("Result", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
