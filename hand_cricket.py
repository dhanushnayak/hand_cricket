
import cv2
import mediapipe as mp
import math
import random
from pillowdrawtable import drawtable
import PIL 
import numpy as np
import cvzone
import time
#table_font = PIL.ImageFont.truetype("arial.tff", 10)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text



def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cvzone.cornerRect(img, (button.pos[0], button.pos[1], button.size[0], button.size[1]), 20, rt=0)
        cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

    return img

class HandCricket:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True, flipType=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)
                myHand["lmList"] = mylmList
                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                                self.mpHands.HAND_CONNECTIONS)
        if draw:
            return allHands, img
        else:
            return allHands

    def fingersUp(self, myHand):
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :return: List of which fingers are up
        """
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        
        if self.results.multi_hand_landmarks:
            fingers = []
            # Thumb
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img=None,draw=True):
        """
        Find the distance between two landmarks based on their
        index numbers.
        :param p1: Point1
        :param p2: Point2
        :param img: Image to draw on.
        :param draw: Flag to draw the output on the image.
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            if draw:
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length, info, img
        else:
            return length, info                                      



data_of_instance ={
    "break_cam":False,
    'score':0,
    "system_score":0,
    "status":"Inactive",
    "total_score":0,
    "game_over":False,
    "mode":'hard'
}

cap = cv2.VideoCapture(0)

def get_image():
    mode =  data_of_instance['mode']
    hcric = HandCricket(detectionCon=0.9, maxHands=2)
    if mode=='hard':decay_rate = 1
    if mode=='medium':decay_rate = 0.5
    if mode=='easy': decay_rate = 0.1
    delay = 100
    while cap.isOpened():
        success, img = cap.read()
        h,w,c = img.shape
        if str(data_of_instance['status']).lower()=='inactive':
            cv2.putText(img,'Please show your hand to cam',(w//6,h//2), cv2.FONT_HERSHEY_SIMPLEX, 1,
                  (153, 0, 153), 2, cv2.LINE_AA, False)
            status_color = (0,0,179)
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        img.flags.writeable = False
        hands = hcric.findHands(img,draw=False)
        pose_result = pose.process(img)
        if len(hands)>0: 
            data_of_instance["status"] = 'Active'
            status_color = (37, 247, 5)
            
        if hands and not data_of_instance['game_over']:
            hand1 = hands[0]
            handType1 = hand1["type"]  # Handtype Left or Right
            fingers1 = hcric.fingersUp(hand1)
            if fingers1[0]==1 and len(hands)==1 and sum(fingers1)!=5:
                data_of_instance["score"] = 6 + sum(fingers1[1:])
            elif len(hands)==1: 
                data_of_instance["score"] = sum(fingers1)
            if len(hands)==2:
                hand2 = hands[1]
                handType2 = hand2["type"]  # Handtype Left or Right
                fingers2 = hcric.fingersUp(hand2)
                data_of_instance["score"] = sum(fingers1)+sum(fingers2)
        
            
        if data_of_instance["status"]=='Active' and not data_of_instance["game_over"]:
            
            lwrist = pose_result.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_WRIST].x * w
            rwrist = pose_result.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_WRIST].x * w
            if rwrist>lwrist: data_of_instance["status"]='Inactive'
                
            delay-=decay_rate
            if delay//10 in [3,2,1]:
                cv2.putText(img,f'{round(delay//10)}',(w//2,h//2), cv2.FONT_HERSHEY_SIMPLEX, 5,
                  (19 ,237, 234), 5, cv2.LINE_AA, False)
            
            
            if round(delay,1)==0.0:  
                data_of_instance["system_score"] = random.randint(1,10)
                score_up = data_of_instance['score']
                delay=100
                
                if data_of_instance["system_score"] == data_of_instance['score']:
                    data_of_instance["game_over"] = True
                    
                if data_of_instance["game_over"] and data_of_instance["total_score"] == 0: data_of_instance["total_score"] = 0
                elif data_of_instance["game_over"]: data_of_instance["total_score"]+=0
                else: data_of_instance["total_score"]+=score_up


            if delay//10 in [10,9] and data_of_instance["total_score"]!=0:
                cv2.putText(img,f'Score = {data_of_instance["score"]}',(w//4,h//2), cv2.FONT_HERSHEY_SIMPLEX, 2,
                  (255, 178, 26), 5, cv2.LINE_AA, False)
                    
                

        #cv2.rectangle(img,(5,5),(int(w*0.35),int(h*0.35)),(230, 230, 230),-1)

        cv2.putText(img, f'Status', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                  status_color, 2, cv2.LINE_AA, False) 
        """cv2.putText(img, f'Hit = {str(score)}', (7,70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                  (153, 0, 153), 2, cv2.LINE_AA, False) 
        cv2.putText(img, f'Score = {str(total_score)}', (7,110), cv2.FONT_HERSHEY_SIMPLEX, 1,
                  (153, 0, 153), 2, cv2.LINE_AA, False) """

        img = PIL.Image.fromarray(img)
        table_score = [(f"Total Score: {data_of_instance['total_score']}",f"Score: {data_of_instance['score']}",f"AI: {data_of_instance['system_score']}")]
        table_draw = drawtable.Drawtable(x=0,y=h-50,xend=w,drawsheet=img,data=table_score,font_size=36,text_stroke_fill=(255,255,255),frame=False,text_color=(163, 24, 222),grid=False,columngrid=False,rowgrid=False,header=False,columnwidth=[0.5,0.3,0.2])
        table_draw.draw_table()
        img= cv2.cvtColor(np.array(img), cv2.COLOR_RGB2RGBA)

        data_of_instance["score"] = 0
        if data_of_instance['game_over'] :

                    cv2.putText(img,f'GAME OVER',(w//4,(h//2)-40), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (46, 0, 255), 3, cv2.LINE_AA, False)
                    cv2.putText(img,f'Score = {data_of_instance["total_score"]}',(w//4,(h//2)+20), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (46, 0, 255), 3, cv2.LINE_AA, False)
                    buttonList = []
                    text_repo = ['RESET','EXIT']
                    for i in range(len(text_repo)):
                            if i==0:size=[230,85]
                            else: size = [165,85]
                            buttonList.append(Button([300 * i + 100, (h//2)+40], text_repo[i],size=size))
                    img = drawAll(img, buttonList)
                    if hands:
                        lmList = hands[0]["lmList"] 
                        if lmList:
                            for button in buttonList:
                                x,y = button.pos
                                w,h = button.size

                                if x < lmList[8][0] < x+w and y < lmList[8][1] < y+h:
                                    cv2.rectangle(img, (x-5, y-5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                                    cv2.putText(img, button.text, (x + 20, y + 65),
                                                cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                                    l, _, _ = hcric.findDistance(lmList[8][:2], lmList[12][:2], img,draw=False)
                                    
                                    # When Clicked
                                    if l < 30:
                                        cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                                        cv2.putText(img, button.text, (x + 20, y + 65),
                                                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                                        #finalText += button.text
                                        if button.text =='EXIT':
                                            data_of_instance["break_cam"] = True
                                            time.sleep(2)
                                        if button.text == "RESET":
                                            data_of_instance["total_score"] = 0
                                            data_of_instance["game_over"] = False
                                            data_of_instance["system_score"] = 0
                                            data_of_instance["score"] = 0
                                            
                                          
                    
        #time.sleep(0.3)
        cv2.imshow('MediaPipe Hands', img)
    
        keys = cv2.waitKey(1) & 0xFF 
        if keys == ord('q') or data_of_instance['break_cam']:
            break
            cap.release()
        elif data_of_instance["break_cam"] or keys == ord('s'):
            print('s is pressed')
            break
            cap.release()

if __name__=='__main__':
    get_image()
        
    

