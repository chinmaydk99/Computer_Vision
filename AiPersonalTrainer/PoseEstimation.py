import cv2
import mediapipe as mp
import time
import math
import numpy as np
class PoseEstimator:
    def __init__(self,static_image_mode=False, model_complexity=1, smooth_landmarks=True,
               enable_segmentation=False, smooth_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity, self.smooth_landmarks,
               self.enable_segmentation, self.smooth_segmentation, self.min_detection_confidence, self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawingSpec = self.mpDraw.DrawingSpec((255,0,0), thickness = 1, circle_radius =1)

    def findPose(self,img,draw = True):
        imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS,
                                           landmark_drawing_spec = self.drawingSpec)
        return img
    
    def findPosition(self,img,draw = True):
        self.landmark_list = []
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id,lm)
                cx, cy  = int(lm.x*w), int(lm.y*h)
                self.landmark_list.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),5,(0,255,0),cv2.FILLED)
        return self.landmark_list
    
    def findAngle(self, img, p1, p2, p3, draw=True):
        # Extracting x and y coordinates for the landmark list
        x1, y1 = self.landmark_list[p1][1:]
        x2, y2 = self.landmark_list[p2][1:]
        x3, y3 = self.landmark_list[p3][1:]

        # Calculating the angle
        angle = math.degrees(math.atan2(y3-y2,x3-x2) - 
                             math.atan2(y1-y2,x1-x2))
        
        # Accounting for negative values
        if angle < 0:
            angle = angle + 360
        
        #print(angle)

        # Verifying that we have obtained the right coordinates
        if draw:
            cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
            cv2.line(img,(x3,y3),(x2,y2),(255,255,255),3)
            cv2.circle(img,(x1,y1), 10, (0,0,255), cv2.FILLED)
            cv2.circle(img,(x1,y1), 15, (0,0,255), 2)
            cv2.circle(img,(x2,y2), 10, (0,0,255), cv2.FILLED)
            cv2.circle(img,(x2,y2), 15, (0,0,255), 2)
            cv2.circle(img,(x3,y3), 10, (0,0,255), cv2.FILLED)
            cv2.circle(img,(x3,y3), 15, (0,0,255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        
        return angle
                        
def main():
    #cap = cv2.VideoCapture('D:\\Courses\\OpenCV\\Projects\\AiPersonalTrainer\\TrainingVideos\\Curl2.mp4')
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    estimator = PoseEstimator()
    while True:
        success, img = cap.read()
        #img = cv2.imread('D:\\Courses\\OpenCV\\Projects\\AiPersonalTrainer\\images\\test2.jpg')
        img = estimator.findPose(img, draw=False)
        landmark_list = estimator.findPosition(img,draw = False)
        
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)),(70,50),
                cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),4)
        cv2.imshow('Video',img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()