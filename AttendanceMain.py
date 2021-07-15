import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path='FaceRecognition/ImagesAttendance'
imgs=[]
classNames=[]
list=os.listdir(path)
print(list)
for cls in list:
    curImg=cv2.imread(f'{path}/{cls}')
    imgs.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

def findEncoding(imgs):
    encodelist=[]
    for img in imgs:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

# ********* FUNCTION FOR MARKING ATTENDANCE ***********
def markAttendance(name):
    with open('FaceRecognition/Attendance.csv','r+') as f:
        datalist=f.readlines()
        #print(datalist)
        namelist=[]
        for line in datalist:
            entry=line.split(',')
            namelist.append(entry[0]) #appends only the name
        if name not in namelist:
            now=datetime.now()
            dstring=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dstring}')

# ******* END OF FUNCTION *******

encodelistknown=findEncoding(imgs)
#print(len(encodelistknown))
print('Encoding complete!')

cap=cv2.VideoCapture(0)
while True:
    succ,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    #camera may capture multiple faces so in order to reduce that use the faceLoc feature:
    facesCurFrame=face_recognition.face_locations(imgS)
    encodesCurFrame=face_recognition.face_encodings(imgS,facesCurFrame)
    #iterate through the encodings and faceCurframe for locations and compare encodings from before:
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):           #use "zip" in order to iterate in the same loop
        matches=face_recognition.compare_faces(encodelistknown,encodeFace)
        faceDis=face_recognition.face_distance(encodelistknown,encodeFace)
        #print(faceDis)
        matchno=np.argmin(faceDis)

        if matches[matchno]:
            name=classNames[matchno].upper()
            #print(name)

            #creating a boundingbox around the image
            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4   #because we reduced the size to 1/4 th before now we multiply by 4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y1-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)     # *******Testing value******


    cv2.imshow("Frame",img)
    if cv2.waitKey(1)==ord('q'):
        break

# For adding someone new just add images in the Images attendance file/folder whatever you choose.
