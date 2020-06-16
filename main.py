#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 20:31:45 2018

@author: shashidhar
"""
# importing face_recognition, OpenCV, numpy packages
import face_recognition
import cv2
import numpy as np


# Loading images
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
nikhilk=face_recognition.load_image_file('nikhil k.jpg')
pranavi=face_recognition.load_image_file('pranavi.jpg')
prasad=face_recognition.load_image_file('prasad.jpg')
prem=face_recognition.load_image_file('prem.jpg')
aditya=face_recognition.load_image_file('aditya vikram.jpg')
santhosh=face_recognition.load_image_file('santhosh.jpg')
vijaysimha=face_recognition.load_image_file('vijaysimha.jpg')
vikram=face_recognition.load_image_file('vikram aditya.jpg')
vivek=face_recognition.load_image_file('vivek.jpg')
anil=face_recognition.load_image_file('anil.jpg')
rohan=face_recognition.load_image_file('rohan.jpg')
rksir=face_recognition.load_image_file('ramakrishna sir.jpg')
snikhil=face_recognition.load_image_file('sainikhil.jpg')
vishal=face_recognition.load_image_file('vishal.jpg')
venko=face_recognition.load_image_file('dvenkatesh.jpg')
psk=face_recognition.load_image_file('praneeth.jpg')



# Encoding images
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
image13_code = face_recognition.face_encodings(nikhilk)[0]
image14_code = face_recognition.face_encodings(pranavi)[0]
image15_code = face_recognition.face_encodings(prasad)[0]
image16_code = face_recognition.face_encodings(prem)[0]
image17_code = face_recognition.face_encodings(aditya)[0]
image18_code = face_recognition.face_encodings(santhosh)[0]
image19_code = face_recognition.face_encodings(vijaysimha)[0]
image20_code = face_recognition.face_encodings(vikram)[0]
image21_code = face_recognition.face_encodings(vivek)[0]
image22_code = face_recognition.face_encodings(anil)[0]
image23_code = face_recognition.face_encodings(rohan)[0]
image24_code = face_recognition.face_encodings(rksir)[0]
image25_code = face_recognition.face_encodings(snikhil)[0]
image26_code = face_recognition.face_encodings(vishal)[0]
image27_code = face_recognition.face_encodings(venko)[0]
image28_code = face_recognition.face_encodings(psk)[0]


known_face_encodings = [image1_code,image2_code,image3_code,image4_code
                        ,image5_code,image6_code,image7_code,image8_code
                        ,image9_code,image10_code,image11_code,image12_code
                        , image13_code, image14_code, image15_code, image16_code
                        , image17_code, image18_code, image19_code, image20_code
                        , image21_code, image22_code, image23_code, image24_code
                        , image25_code, image26_code, image27_code, image28_code]

# names of known faces
known_face_names = ['bhanu prakash', 'charan preet', 'gowtam', 'hrushikesh', 
                    'ismail', 'sai kowshik', 'saketh' ,'shashidhar', 'shivakiran' 
                    , 'sravan', 'sujith', 'venkatesh m', 'nikhil k', 'pranavi', 
                    'prasad', 'prem', 'aditya vikram', 'santhosh', 'vijaysimha',
                    'vikram aditya', 'vivek', 'anil', 'rohan','RamKrishna'
                    ,'SaiNikhil', 'Vishal', 'Venko', 'Psk']


# starting webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    # converting BGR(openCV) to RGB(useful for face_recognition)
    rgb_frame = frame[:, :, ::-1]
    # locating faces on present frame
    locations = face_recognition.face_locations(rgb_frame)
    # Encoding faces on present frame
    face_codes = face_recognition.face_encodings(rgb_frame, locations)
    
    for (x, y, w, h), face_encoding in zip(locations, face_codes):
        # finding the distance between present face encoding with faces present in database
        check_mat=face_recognition.face_distance(known_face_encodings,face_encoding)
        index=np.argmin(check_mat)
        diff=min(check_mat)
        # predicting person's name
        if diff>0.45:
            name='no data'
        else:
            name=known_face_names[index]   
        # Drawing a rectangle around the detected face   
        cv2.rectangle(frame, (h, x), (y, w), (255,0,0), 2)
        # Putting text of the person's name on frame 
        cv2.putText(frame, name, (h-6,x), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,255,0), 2)
    # Displaying the frame    
    cv2.imshow('Video', frame)
    # press 'Q' to close the webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()