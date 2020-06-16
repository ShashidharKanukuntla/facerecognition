#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 20:14:22 2018

@author: shashidhar
"""

import cv2
import numpy as np
import face_recognition

#Loading images
bhanu=face_recognition.load_image_file('bhanu prakash.jpg')
charan_preet=face_recognition.load_image_file('charan preet.jpg')
gowtam=face_recognition.load_image_file('gowtam.jpg')
hrushi=face_recognition.load_image_file('hrushikesh2.jpg')
ismail=face_recognition.load_image_file('ismail.jpg')
kowshik=face_recognition.load_image_file('sai kowshik.jpg')
saketh=face_recognition.load_image_file('saketh.jpg')
shashidhar=face_recognition.load_image_file('shashidhar.jpg')
shiva=face_recognition.load_image_file('shiva kiran.jpg')
sravan=face_recognition.load_image_file('sravan.jpg')
sujith=face_recognition.load_image_file('sujith.jpg')
venky=face_recognition.load_image_file('venkatesh m.jpg')


image1_code = face_recognition.face_encodings(bhanu)[0]
image2_code = face_recognition.face_encodings(charan_preet)[0]
image3_code = face_recognition.face_encodings(gowtam)[0]
image4_code = face_recognition.face_encodings(hrushi)[0]
image5_code = face_recognition.face_encodings(ismail)[0]
image6_code = face_recognition.face_encodings(kowshik)[0]
image7_code = face_recognition.face_encodings(saketh)[0]
image8_code = face_recognition.face_encodings(shashidhar)[0]
image9_code = face_recognition.face_encodings(shiva)[0]
image10_code = face_recognition.face_encodings(sravan)[0]
image11_code = face_recognition.face_encodings(sujith)[0]
image12_code = face_recognition.face_encodings(venky)[0]

known_codes = [image1_code,image2_code,image3_code,image4_code,image5_code,image6_code,image7_code,image8_code,image9_code,image10_code,image11_code,image12_code]

face_names = ['bhanu prakash', 'charan preet', 'gowtam', 'hrushikesh', 'ismail', 'sai kowshik', 'saketh' ,'shashidhar', 'shivakiran' , 'sravan', 'sujith', 'venkatesh m']

def getname(known_codes,new_code,names):
    check_mat=face_recognition.face_distance(known_codes,new_code)
    index=np.argmin(check_mat)
    diff=min(check_mat)
    if diff>0.45:
        name='no data'
    else:
        name=names[index]
    return name    


# Doing some Face Recognition with the webcam
    

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect(gray,frame,known_faces,face_names):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    locations=face_recognition.face_locations(frame)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        prescode=face_recognition.face_encodings(frame[y:y+h,x:x+w])
        if len(prescode)>0:
            testname=[]
            testcode=[]
            for i in range(0,len(locations)-1):
                testcode[i]=test[i]
                testname[i]=getname(known_codes,testcode[i],face_names)
                cv2.putText(frame, testname[i] , (x-10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else :
            print('no face found')
            continue
    return frame

videocap = cv2.VideoCapture(0)
while True:
    ret,frames = videocap.read()
    frame = frames[:, :, ::-1]
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    canvas = detect(gray,frame,known_codes,face_names)
    cv2.imshow("Video", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
videocap.release()
cv2.destroyAllWindows()



imgnew=face_recognition.load_image_file('venky and venko.jpg')
locations=face_recognition.face_locations(imgnew)
testcode=[]
testname=[]
for i in range(0,len(locations)-1):
    testcode(i)=face_recognition.face_encodings(imgnew)[i]
    testname[i]=getname(known_codes,testcode[i],face_names)
    
testcode= face_recognition.face_encodings(imgnew)
testcode0=testcode[0]
testcode1=testcode[1]
testname=getname(known_codes,testcode1,face_names)