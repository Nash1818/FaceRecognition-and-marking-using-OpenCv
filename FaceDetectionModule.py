import cv2
import mediapipe as mp
import time

from mediapipe.python.solutions import face_detection

class FaceDetector():
    def __init__(self,minDetectionCon=0.5):

        self.minDetectionCon=minDetectionCon
        self.mpFace=mp.solutions.face_detection
        self.mpDraw=mp.solutions.drawing_utils
        self.faceDetection=self.mpFace.FaceDetection()  
        # 0.5 is the min Detection Condidence we change it to remove the false Positives




    def findFaces(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.faceDetection.process(imgRGB)
        #print(self.results)
        bboxs=[]

        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                #mpDraw.draw_detection(img,detection)       # Drawing using default function of mediapipe
                #print(id,detection)
                #print(detection.score)
                #print(detection.location_data.relative_bounding_box)
                h,w,c=img.shape
                bboxC=detection.location_data.relative_bounding_box
                bbox=int(bboxC.xmin*w),int(bboxC.ymin*h),\
                int(bboxC.width*w),int(bboxC.height*h)
                bboxs.append([id,bbox,detection.score])
                self.fancyDraw(img,bbox)
                if draw:
                    img=self.fancyDraw(img,bbox)
                    #cv2.rectangle(img,bbox,(255,0,255),2)      # Drawing using cv2 with the values
        return img,bboxs

    def fancyDraw(self,img,bbox,l=30,t=10):
        x,y,w,h= bbox
        x1,y1=x+w,y+h

        cv2.rectangle(img,bbox,(255,0,255),2)
        #Top Left x,y
        cv2.line(img,(x,y),(x+l,y),(255,0,255),t)  #For drawing thick line above
        cv2.line(img,(x,y),(x,y+l),(255,0,255),t)  #For drawing thick line below
        #Top right x1,y
        cv2.line(img,(x1,y),(x1-l,y),(255,0,255),t)  #For drawing thick line above
        cv2.line(img,(x1,y),(x1,y+l),(255,0,255),t)  #For drawing thick line below

        #Bottom Left x,y1
        cv2.line(img,(x,y1),(x+l,y1),(255,0,255),t)  #For drawing thick line above
        cv2.line(img,(x,y1),(x,y1-l),(255,0,255),t)  #For drawing thick line below
        #Bottom right x1,y1
        cv2.line(img,(x1,y1),(x1-l,y1),(255,0,255),t)  #For drawing thick line above
        cv2.line(img,(x1,y1),(x1,y1-l),(255,0,255),t)  #For drawing thick line below

        return img
        
def main():
    cap=cv2.VideoCapture(0)
    ptime=0
    detector=FaceDetector()
    while True:
        succ,img=cap.read()
        img,bboxs=detector.findFaces(img)
        
        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime
        cv2.putText(img,str(int(fps)),(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)
        cv2.imshow("Video",img)

        if cv2.waitKey(1)==ord('q'):
            break

if __name__=="__main__":
    main()