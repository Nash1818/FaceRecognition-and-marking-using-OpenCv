import cv2
import numpy as np
import face_recognition

#Gaining both the images for the test
imgElon=face_recognition.load_image_file('FaceRecognition/elonbasic.jpg')
imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgtest=face_recognition.load_image_file('FaceRecognition/elontest.jpg')
imgtest=cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

#Obtaining the actual image models
faceLoc=face_recognition.face_locations(imgElon)[0]
encodeElon=face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

#Obtaining the test model
faceLocTest=face_recognition.face_locations(imgtest)[0]
encodetest=face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

#Testing the encodings using SVM model for similarity 
results=face_recognition.compare_faces([encodeElon],encodetest)
faceDis=face_recognition.face_distance([encodeElon],encodetest)

#Printing results as True or False
print(results,faceDis)
cv2.putText(imgtest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

cv2.imshow('ElonMusk',imgElon)
cv2.imshow('ElonTest',imgtest)
cv2.waitKey(0)