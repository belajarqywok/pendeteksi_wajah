#!/bin/env python
import cv2 as visionLiblary
import numpy as numerialLib

videoCam = visionLiblary.VideoCapture(0)

face = visionLiblary.CascadeClassifier('wajah.xml')
eye = visionLiblary.CascadeClassifier('mata.xml')

while True:
    cond, frame = videoCam.read()

    gray = visionLiblary.cvtColor(frame, visionLiblary.COLOR_BGR2GRAY)
    muka = face.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in muka:
        visionLiblary.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 5)

        roi_warna = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]
        mata = eye.detectMultiScale(roi_gray)
        for (mx,my,mw,mh)in mata:
            visionLiblary.rectangle(roi_warna, (mx,my), (mx+mw, my+mh), (255,255,0), 2)

    visionLiblary.imshow('gabut', frame)

    k = visionLiblary.waitKey(1) & 0xff
    if k == ord('q'):
        break

videoCam.release()
visionLiblary.destroyAllWindows()
