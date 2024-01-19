import cv2
import mediapipe as mp
import time

class HandDetector:

    def __init__(self,mode = False,maxHands =2 , model_complexity = 1, detectionCon = 0.5, trackCon =0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.model_complexity = model_complexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity,self.detectionCon, self.trackCon) 
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec((0,255,0),thickness=2,circle_radius = 2)

    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS,(0,255,0),3)

        return img
    
    def findPosition(self,img, handNumber = 0, draw = True, req_id=None):
        
        landmark_list = [] #List of all landmarks
        if self.results.multi_hand_landmarks: #Checking if hands are detected
            myHand = self.results.multi_hand_landmarks[handNumber]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                landmark_list.append([id,cx,cy])
                
                # if id == req_id:
                #     cv2.circle(img,(cx,cy),5,(255,0,155),cv2.FILLED)

                if draw:
                    cv2.circle(img,(cx,cy),5,(0,0,255),cv2.FILLED)
        
        return landmark_list
            
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read() 
        img = detector.findHands(img)
        landmark_list = detector.findPosition(img)

        if landmark_list:
            print(landmark_list[4])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime =cTime
        cv2.putText(img,str(int(fps)),(18,78),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),4)

        cv2.imshow('Video',img)
        #cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
