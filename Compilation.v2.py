#COMPILATION FOR FINALISED UTP FYP CODING
#MOHD ISKANDAR BIN MD JOHADI
#UTP ID 18002410

#IMPORTS
from djitellopy import tello
import cv2 as cv
import mediapipe as mp
import time
import utils, math
import numpy as np
import random


#GLOBAL VARIABLES
CEF_COUNTER = 0
TOTAL_BLINKS = 0

#def= True
tello_cc= False
control_center = True
#def= False
mainprogram = True
handmode = False
facemode = True
eyemode = False
fControl = True
flight = False
loading = 100
speed = 40
counter = 0
counterTakeoff = 0
counterLand = 0
ptime = 0
tf_w = 360
tf_h = 240
speech = 0

#GLOBAL CONTANTS
CLOSED_EYES_FRAME = 50 #50 frames eyes needed to close before accept as hard blink
# face bounder indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
# lips indices for Landmarks
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
        37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
## left eyes indices
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
L_H_LEFT = [33]
L_H_RIGHT = [133]
## right eyes indices
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
R_H_RIGHT = [263]
R_H_LEFT = [362]
## iris indices
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]
## nose indices
NOSE_CENTER = [4]
i = 0
text = "Hi there!"
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)

#GESTURE FUNCTIONS
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance

def piece_positioner(center_point, point_r_u, point_l_d, position):
    center_to_right_dist = euclaideanDistance(center_point,point_r_u)
    total_distance = euclaideanDistance(point_r_u,point_l_d)
    ratio = center_to_right_dist/total_distance
    piece_position = ""
    if position == "hor_right":
        if ratio< 0.48:
            piece_position = "RIGHT"
        elif ratio>0.48 and ratio < 0.57:
            piece_position = "CENTER"
        else:
            piece_position = "LEFT"
    elif position == "hor_left":
        if ratio< 0.48:
            piece_position = "LEFT"
        elif ratio>0.48 and ratio < 0.57:
            piece_position = "CENTER"
        else:
            piece_position = "RIGHT"
    elif position == "hor_head":
        if ratio< 0.35:
            piece_position = "LEFT"
        elif ratio>0.35 and ratio < 0.65:
            piece_position = "CENTER"
        else:
            piece_position = "RIGHT"
    else:
        if ratio < 0.45:
            piece_position = "UP"
        elif ratio > 0.45 and ratio < 0.55:
            piece_position = "MIDDLE"
        else:
            piece_position = "DOWN"


    return piece_position, ratio

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):

    # Right eyes
    # horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes
    # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # LEFT_EYE
    # horizontal line
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    ##prevent ZeroDivisionError: float division by zero
    if rvDistance == 0: rvDistance=0.00001
    if lvDistance == 0: lvDistance=0.00001

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    ratio = (reRatio + leRatio) / 2
    return ratio

def landmarksDetection(img, results,color, thickness, draw=False):
    img_height, img_width = img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height))
                  for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, thickness, color, -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks
    return mesh_coord

class handLandmarkDetector:
    def __init__(self, mode=False, model_complexity=1, maxHands=2, detectionConf=0.5, trackConf=0.5):
        self.mode = mode
        self.modelComplex = model_complexity  # rex added init 14 March
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionConf,
                                         self.trackConf)  # self.modelComplex added f
        self.mp_draw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        #img = cv.flip(img, 1)
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            # Draw hand to image
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return img

    def drawFingerPoint(self, img, drawLeft=True, drawRight=True, finger=8):  # finger id 8 is index finger
        # get handedness
        if self.results.multi_hand_landmarks:
            for id_hnd, hnd in enumerate(self.results.multi_handedness):
                hnd_name = hnd.classification[0].label
                hand = self.results.multi_hand_landmarks[id_hnd]

                h, w, c = img.shape

                for id, lm in enumerate(hand.landmark):
                    # draw left finger
                    if drawLeft:
                        if id == finger and hnd_name == 'Left':
                            ind_finger_l_x = int(lm.x * w)
                            ind_finger_l_y = int(lm.y * h)
                            cv.circle(img, (int(ind_finger_l_x), int(ind_finger_l_y)), 25, GREEN
                                      , cv.FILLED)
                            cv.putText(img, hnd_name, (int(ind_finger_l_x)-25, int(ind_finger_l_y)-30), cv.FONT_HERSHEY_SIMPLEX, 1,
                                       GREEN
                                       , 3)  # draws left to img if left hand is detected

                    # draw right finger
                    if drawRight:
                        if id == finger and hnd_name == 'Right':
                            ind_finger_r_x = int(lm.x * w)
                            ind_finger_r_y = int(lm.y * h)
                            cv.circle(img, (int(ind_finger_r_x), int(ind_finger_r_y)), 25, RED, cv.FILLED)
                            cv.putText(img, hnd_name, (int(ind_finger_r_x)-35, int(ind_finger_r_y)-30), cv.FONT_HERSHEY_SIMPLEX, 1,
                                       RED, 3)  # draws left to img if left hand is detected

                    try:
                        return [(ind_finger_l_x, ind_finger_l_y), (ind_finger_r_x, ind_finger_r_y)]
                    except:
                        pass


def in_circle(center_x, center_y, radius, coords):
    x, y = coords
    square_dist = (center_x - x) ** 2 + (center_y - y) ** 2
    return square_dist <= radius ** 2


#MAIN CODE
mp_face_mesh = mp.solutions.face_mesh
hand_detector = handLandmarkDetector()
key = ord("p")

##CAMERA INIIALIZATION
cap = cv.VideoCapture(0)

face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5,
    )

if tello_cc:
    ################# set up Tello ##################
    myTello = tello.Tello()
    myTello.connect()
    print(myTello.get_battery())
    myTello.streamon()

while True:

    if tello_cc:
        ###TELLO STREAM CAPTURE
        telloframe = myTello.get_frame_read().frame

    ret, vidframe = cap.read()
    if not ret:
        break


    frame = cv.flip(vidframe, 1)  # mirror to user perception
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # use rgb value for frame
    frame = cv.resize(frame, None, fx=1.8, fy=1.8,
                         interpolation=cv.INTER_CUBIC)  # resize frame for better view
    img_h, img_w = frame.shape[:2]  # take resized frame dimension
    results = face_mesh.process(rgb_frame)  # process frame

    if results.multi_face_landmarks:
        mesh_points = np.array(
            [
                np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                for p in results.multi_face_landmarks[0].landmark
            ]
        )

    ###SETTING CUSTOMIZATION
    if control_center:
        # create new frame for joystick image (control center mode)
        CC_COLOR = (0, 0, 0)  # hand mode joystick color become black
        frame = np.zeros((img_h, img_w, 3), np.uint8)
        frame[:] = (255, 255, 255)
        #cv.circle(frame, mesh_points[NOSE_CENTER][0], 25, GREEN
        # , cv.FILLED)

        if tello_cc:
            telloframe = cv.resize(telloframe, (tf_w, tf_h))
            #frame[img_h - 240:img_h, img_w - 360 * 2 - 50:img_w - 360 - 50] = telloframe
    # if webcam mode
    else:
        CC_COLOR = (0, 255, 255)  # turns black into yellow for better visibility
        #frame = vidframe
        frame = frame
        if tello_cc:
            telloframe = cv.resize(telloframe, (img_w, img_h))
            #frame[0:img_h, 0:img_w] = telloframe
        #cv.circle(frame, mesh_points[NOSE_CENTER][0], 25, GREEN
        # , cv.FILLED)

    cv.circle(frame, mesh_points[NOSE_CENTER][0], 25, GREEN, cv.FILLED)
    cv.circle(frame, mesh_points[454], 1, GREEN, cv.LINE_AA)
    cv.circle(frame, mesh_points[234], 1, GREEN, cv.LINE_AA)
    cv.circle(frame, mesh_points[10], 1, GREEN, cv.LINE_AA)
    cv.circle(frame, mesh_points[0], 1, GREEN, cv.LINE_AA)
    cv.circle(frame, mesh_points[17], 1, GREEN, cv.LINE_AA)
    cv.circle(frame, mesh_points[152], 1, GREEN, cv.LINE_AA)


    landmarksDetection(frame, results, CC_COLOR, 2, True)
    cv.polylines(frame, [np.array([mesh_points[p] for p in FACE_OVAL], dtype=np.int32)], True, CC_COLOR, 1,
                 cv.LINE_AA)

    # draw control activation circles
    cv.putText(frame, 'Hand Mode', (int(img_w * 0.8)-80, int(img_h * 0.5)-60), cv.FONT_HERSHEY_SIMPLEX, 1,
               CC_COLOR, 2)
    cv.circle(frame, (int(img_w * 0.8), int(img_h * 0.5)), 50, CC_COLOR, 3) ##hand mode
    cv.putText(frame, 'Face & Eye Mode', (int(img_w * 0.2)-130, int(img_h * 0.5)-60), cv.FONT_HERSHEY_SIMPLEX, 1,
               CC_COLOR, 2)
    cv.circle(frame, (int(img_w * 0.2), int(img_h * 0.5)), 50, CC_COLOR, 3) ##face mode

    # cv.putText(frame, 'To activate Tello hand controller', (int(img_w * 0.3) - 45, 300), cv.FONT_HERSHEY_SIMPLEX, 1, CC_COLOR, 2)
    # cv.putText(frame, 'move both index fingers', (int(img_w * 0.35) - 40, 340), cv.FONT_HERSHEY_SIMPLEX, 1,CC_COLOR, 2)
    # cv.putText(frame, 'into the boxes above', (int(img_w * 0.38) - 43, 380), cv.FONT_HERSHEY_SIMPLEX, 1, CC_COLOR, 2)
    cv.putText(frame, 'To activate gesture tracking program', (int(img_w * 0.3) - 45, 30), cv.FONT_HERSHEY_SIMPLEX, 1,
               CC_COLOR, 2)
    cv.putText(frame, 'move your green nose cursor', (int(img_w * 0.35) - 50, 70), cv.FONT_HERSHEY_SIMPLEX, 1,
               CC_COLOR, 2)
    cv.putText(frame, 'into the choice circles', (int(img_w * 0.38) - 40, 110), cv.FONT_HERSHEY_SIMPLEX, 1, CC_COLOR,
               2)

    i = random.randint(0, 444)
    #print(i)
    ###MISC
    speech +=1
    if speech == 50:
        i = random.randint(0, 15)
        if i == 0: text = "Hmm"
        elif i == 1: text = "Should we check this?"
        elif i == 3: text = "The other one, maybe?"
        elif i == 5: text = "Let's  do this one?"
        elif i == 7: text = "Maybe the other one?"
        else: text = ""
        speech = 0
    cv.putText(frame, f'{text}', (mesh_points[FACE_OVAL][0]), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 255),1, cv.LINE_AA)

    ###GET FPS
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv.putText(frame, f'{fps:.2f}', (int(img_w * 1) - 120, 30), cv.FONT_HERSHEY_PLAIN, 2, GREEN
               , 1,
               cv.LINE_AA)

    ###PROGRAM MODE CHECKER
    try:
        if in_circle(int(img_w * 0.8), int(img_h * 0.5), 50, mesh_points[NOSE_CENTER][0]):
            counter += 1
            print(counter)
            cv.putText(frame, f'Program starting in.. {loading - counter}', (int(img_w * 0.38) - 55, 160),
                       cv.FONT_HERSHEY_PLAIN, 2, GREEN
                       , 1, cv.LINE_AA)
            if counter == loading:
                mainprogram = not mainprogram
                handmode = not handmode
                print('Hand Mode activated', fControl)
                cv.destroyAllWindows()
        elif in_circle(int(img_w * 0.2), int(img_h * 0.5), 50, mesh_points[NOSE_CENTER][0]):
            counter += 1
            cv.putText(frame, f'Program starting in.. {loading - counter}', (int(img_w * 0.38) - 55, 160),
                       cv.FONT_HERSHEY_PLAIN, 2, GREEN
                       , 1, cv.LINE_AA)
            print(counter)
            if counter == loading:
                mainprogram = not mainprogram
                facemode = not facemode
                print('Face Mode activated', fControl)
                cv.destroyAllWindows()
        else:
            counter = 0
    except:
        pass

    ##HAND PROGRAM LOOP
    while handmode and mainprogram:

        ret, vidframe = cap.read()
        if not ret:
            break

        vidframe = cv.flip(vidframe, 1) #mirror to user perception
        rgb_frame = cv.cvtColor(vidframe, cv.COLOR_BGR2RGB) #use rgb value for frame
        vidframe = cv.resize(vidframe, None, fx=1.8, fy=1.8, interpolation=cv.INTER_CUBIC) #resize frame for better view
        img_h, img_w = vidframe.shape[:2] #take resized frame dimension
        results = face_mesh.process(rgb_frame) #process frame

        if tello_cc:
            ###TELLO STREAM CAPTURE
            telloframe = myTello.get_frame_read().frame
            # telloframe = cv.resize(telloframe, (tf_w,tf_h))
            # telloframe = cv.resize(frame,(tf_w,tf_h))
            #telloframe = cv.resize(telloframe, (img_w, img_h))

        ###SETTING CUSTOMIZATION
        if control_center:
            # create new frame for joystick image (control center mode)
            CC_COLOR = (0, 0, 0) # hand mode joystick color become black
            frame = np.zeros((img_h, img_w, 3), np.uint8)
            frame[:] = (255, 255, 255)
            vidframe = hand_detector.findHands(vidframe, draw=True)
            fingerLs = hand_detector.drawFingerPoint(frame)

            if tello_cc:
                telloframe = cv.resize(telloframe, (tf_w, tf_h))
                frame[img_h - 240:img_h, img_w - 360 * 2 -50:img_w - 360 -50] = telloframe
        # if control center mode is set to false
        else:
            CC_COLOR = (0, 255, 255)  # turns black into yellow for better visibility
            frame = vidframe
            frame = hand_detector.findHands(frame, draw=True)
            if tello_cc:
                telloframe = cv.resize(telloframe, (img_w, img_h))
                frame[0:img_h, 0:img_w] = telloframe
            fingerLs = hand_detector.drawFingerPoint(frame)

        speed = 40

        ###GET FPS
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv.putText(frame, f'{fps:.2f}', (int(img_w*1)-120, 30), cv.FONT_HERSHEY_PLAIN, 2, GREEN
                   , 1, cv.LINE_AA)

        ###draw control activation circles
        cv.circle(frame, (int(img_w * 0.4), int(img_h * 0.1)), 25, CC_COLOR, 3)
        cv.circle(frame, (int(img_w * 0.6), int(img_h * 0.1)), 25, CC_COLOR, 3)

        ###CONTROL MODE CHECKER
        try:
            if in_circle(int(img_w * 0.4), int(img_h * 0.1), 25, fingerLs[0]) and in_circle(int(img_w * 0.6),int(img_h * 0.1), 25,fingerLs[1]):
                counter += 1
                if counter == 30:
                    fControl = not fControl
                    print('Control activated', fControl)
            else:
                counter = 0
        except:
            pass


        ###FLIGHT CONTROL SECTION
        if fControl:
            cv.putText(frame, 'CONTROL ACTIVATED', (int(img_w * 0.4) - 40, 30), cv.FONT_HERSHEY_SIMPLEX, 1,CC_COLOR, 3)
            #cv.putText(frame, 'CONTROL DEACTIVATED', (int(img_w * 0.4) - 60, 30), cv.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 255), 3)

            # left joystick
            cv.circle(frame, (int(img_w * 0.3), int(img_h * 0.45)), 125, CC_COLOR, 15)
            cv.putText(frame, str('Rotation'), (int(img_w * 0.07), int(img_h * 0.475)), cv.FONT_HERSHEY_SIMPLEX, 1,
                       CC_COLOR, 2)
            cv.putText(frame, str('Direction'), (int(img_w * 0.245), int(img_h * 0.7) - 40), cv.FONT_HERSHEY_SIMPLEX, 1,
                       CC_COLOR, 2)
            # right joystick
            cv.circle(frame, (int(img_w * 0.7), int(img_h * 0.45)), 125, CC_COLOR, 15)
            cv.putText(frame, str('Yaw'), (int(img_w * 0.83), int(img_h * 0.475)), cv.FONT_HERSHEY_SIMPLEX, 1,
                       CC_COLOR, 2)
            cv.putText(frame, str('Height'), (int(img_w * 0.665), int(img_h * 0.7) - 40), cv.FONT_HERSHEY_SIMPLEX, 1,
                       CC_COLOR, 2)
            # land drone circle
            cv.circle(frame, (int(img_w * 0.4), int(img_h * 0.25)), 25, CC_COLOR, 2)
            cv.putText(frame, str('Land'), (int(img_w * 0.4) - 40, int(img_h * 0.25) - 40), cv.FONT_HERSHEY_SIMPLEX, 1,
                       CC_COLOR, 2)
            # takeoff drone circle
            cv.circle(frame, (int(img_w * 0.6), int(img_h * 0.25)), 25, CC_COLOR, 2)
            cv.putText(frame, str('Takeoff'), (int(img_w * 0.6) - 60, int(img_h * 0.25) - 40), cv.FONT_HERSHEY_SIMPLEX,
                       1, CC_COLOR, 2)

            ####TRY LAND/TAKEOFF COMMAND
            try:
                # Drone Landing
                if in_circle(int(img_w * 0.4), int(img_h * 0.25), 25, fingerLs[0]) or in_circle(int(img_w * 0.4),int(img_h * 0.25), 25,fingerLs[1]):
                    counterLand += 1
                    print(counterLand)
                    cv.putText(frame, f'Landing Drone in.. {20 - counterLand}', (int(img_w * 0.38) - 40, 150),
                               cv.FONT_HERSHEY_PLAIN, 2, GREEN)
                    if counterLand == 20:
                        if tello_cc:
                            myTello.land()
                            print('Tello Landing!!!!')
                        flight = not flight

                elif in_circle(int(img_w * 0.6), int(img_h * 0.25), 25, fingerLs[0]) or in_circle(int(img_w * 0.6), int(img_h * 0.25), 25, fingerLs[1]):
                    counterTakeoff += 1
                    print(counterTakeoff)
                    cv.putText(frame, f'Drone Takeoff in.. {20 - counterTakeoff}', (int(img_w * 0.38) - 40, 150),
                               cv.FONT_HERSHEY_PLAIN, 2, GREEN)
                    if counterTakeoff == 20:
                        if tello_cc:
                            myTello.takeoff()
                            print('Tello Takeoff!!!!')
                        flight = not flight

                else:
                    counterTakeoff = 0
                    counterLand = 0
                print(f'flight {flight}')
            except:
                pass

            ####TRY DIRECTION/VELOCITY COMMAND
            try:
                if in_circle(int(img_w * 0.3), int(img_h * 0.45), 125, fingerLs[0]) and in_circle(int(img_w * 0.7), int(img_h * 0.45), 125,  fingerLs[1]) and flight:

                    ## left joystick
                    # left right
                    if fingerLs[0][0] > int(img_w * 0.3):
                        lr = -((int(img_w * 0.3 - fingerLs[0][0])) / 125)
                    else:
                        lr = -(int(img_w * 0.3 - fingerLs[0][0])) / 125
                    # forward backward
                    if fingerLs[0][1] > int(img_h * 0.45):
                        fb = (int(img_h * 0.45 - fingerLs[0][1])) / 125
                    else:
                        fb = (int(img_h * 0.45 - fingerLs[0][1])) / 125

                    ## right joystick
                    # yaw velocity
                    if fingerLs[1][0] > int(img_w * 0.7):
                        yv = -((int(img_w * 0.7 - fingerLs[1][0])) / 125)
                    else:
                        yv = -(int(img_w * 0.7 - fingerLs[1][0])) / 125
                    # up down
                    if fingerLs[1][1] > int(img_h * 0.45):
                        ud = (int(img_h * 0.45 - fingerLs[1][1])) / 125
                    else:
                        ud = (int(img_h * 0.45 - fingerLs[1][1])) / 125

                    # send rc to tello
                    if tello_cc:
                        myTello.send_rc_control(int(lr * speed), int(fb * speed), int(ud * speed), int(yv * speed))
                    print('left idx F: ', (int(lr * speed), int(fb * speed)), 'right idx F: ',(int(ud * speed), int(yv * speed)))

                #####SAFETY PRECAUTION SECTION
                else:
                    if tello_cc:
                        myTello.send_rc_control(0, 0, 0, 0)
                    cv.circle(frame, (int(img_w * 0.3), int(img_h * 0.45)), 25, CC_COLOR, cv.FILLED)
                    cv.circle(frame, (int(img_w * 0.7), int(img_h * 0.45)), 25, CC_COLOR, cv.FILLED)
            except:
                cv.circle(frame, (int(img_w * 0.3), int(img_h * 0.45)), 25, CC_COLOR, cv.FILLED)
                cv.circle(frame, (int(img_w * 0.7), int(img_h * 0.45)), 25, CC_COLOR, cv.FILLED)
                pass
            '''
            ####OVERLAY TELLO FRAME
            try:
                frame[img_h-240:img_h, img_w-360*2:img_w-360] = telloframe
            except:
                pass
            '''
        else:
            cv.putText(frame, 'CONTROL DEACTIVATED', (int(img_w * 0.4) - 60, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            cv.putText(frame, 'To activate Tello hand controller', (int(img_w * 0.3) - 45, 300), cv.FONT_HERSHEY_SIMPLEX, 1, CC_COLOR, 2)
            cv.putText(frame, 'move both index fingers', (int(img_w * 0.35) - 40, 340), cv.FONT_HERSHEY_SIMPLEX, 1, CC_COLOR, 2)
            cv.putText(frame, 'into the boxes above', (int(img_w * 0.38) - 43, 380), cv.FONT_HERSHEY_SIMPLEX, 1, CC_COLOR, 2)
            '''
            try:
                frame[img_h-tf_h:img_h, img_w-tf_w*2:img_w-tf_w] = telloframe
            except:
                pass
            '''
        if not flight:
            cv.putText(frame, f'Drone is not flying', (img_w - 180, 45), cv.FONT_HERSHEY_PLAIN, 1, RED, 1, cv.LINE_AA)
            #landmarksDetection(frame, results, CC_COLOR, 1, True)
            # cv.polylines(frame, [np.array([mesh_points[p] for p in FACE_OVAL], dtype=np.int32)], True, CC_COLOR, 1,
            #            cv.LINE_AA)

        if control_center:
            cv.imshow("Hand Mode - Control Center", frame)
        else:
            cv.imshow("Hand Mode - Webcam ", frame)


        key = cv.waitKey(1)

        ###CC TOGGLE CHECKER
        if key == ord("c") or key == ord("C"):
            control_center = not control_center
            cv.destroyAllWindows()

        ###PROGRAM BREAK CHECKER
        if key == ord("q") or key == ord("Q"):
            mainprogram = not mainprogram
            handmode = not handmode
            ###SAFETY PRECAUTION SECTION
            if flight:
                if tello_cc:
                    myTello.land()
                    handmode = handmode
                flight = not flight
                if fControl: fControl = not fControl
            cv.destroyAllWindows()
            break

    ##FACE AND EYE MODE PROGRAM
    while (facemode or eyemode) and mainprogram:
        mouth_open = False
        eye_open = False

        ret, vidframe = cap.read()
        if not ret:
            break

        if tello_cc:
            ###TELLO STREAM CAPTURE
            telloframe = myTello.get_frame_read().frame

        frame = cv.flip(vidframe, 1)  # mirror to user perception
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # use rgb value for frame
        frame = cv.resize(frame, None, fx=1.8, fy=1.8,
                          interpolation=cv.INTER_CUBIC)  # resize frame for better view
        img_h, img_w = frame.shape[:2]  # take resized frame dimension
        results = face_mesh.process(rgb_frame)  # process frame

        if results.multi_face_landmarks:
            mesh_points = np.array(
                [
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in results.multi_face_landmarks[0].landmark
                ]
            )

        gap_dist = euclaideanDistance(mesh_points[0], mesh_points[17])
        if gap_dist >= 55 : mouth_open = not mouth_open

        eye_dist2 = euclaideanDistance(mesh_points[159], mesh_points[145])
        eye_dist1 = euclaideanDistance(mesh_points[386], mesh_points[374])
        eyedist = (eye_dist1 + eye_dist2) / 2
        if eyedist > 21: eye_open = not eye_open
        #print(f' eye open is {eye_open}')

        ###SETTING CUSTOMIZATION
        if control_center:
            # create new frame for joystick image (control center mode)
            CC_COLOR = (0, 0, 0)  # hand mode joystick color become black
            frame = np.zeros((img_h, img_w, 3), np.uint8)
            frame[:] = (255, 255, 255)
            # cv.circle(frame, mesh_points[NOSE_CENTER][0], 25, GREEN
            # , cv.FILLED)

            if tello_cc:
                telloframe = cv.resize(telloframe, (tf_w, tf_h))
                frame[img_h - 240:img_h, img_w - 360 * 2 - 50:img_w - 360 - 50] = telloframe
        # if webcam mode
        else:
            CC_COLOR = (0, 255, 255)  # turns black into yellow for better visibility
            # frame = vidframe
            frame = frame
            if tello_cc:
                telloframe = cv.resize(telloframe, (img_w, img_h))
                frame[0:img_h, 0:img_w] = telloframe

            # cv.circle(frame, mesh_points[NOSE_CENTER][0], 25, GREEN
            # , cv.FILLED)

        ###GET FPS
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv.putText(frame, f'{fps:.2f}', (int(img_w*1)-120, 30), cv.FONT_HERSHEY_PLAIN, 2, GREEN
                   , 1, cv.LINE_AA)

        ###BLINK CHECKER & FRAMING
        blink_ratio = blinkRatio(vidframe, mesh_points, RIGHT_EYE, LEFT_EYE) ##alwasy use vidframe for checking eye
        if blink_ratio > 5.5: ##
            CEF_COUNTER += 1
            cv.putText(frame, f'Blink ', (240, 30), cv.FONT_HERSHEY_PLAIN, 2, RED
                       , 1, cv.LINE_AA)

        else:
            if CEF_COUNTER > CLOSED_EYES_FRAME: ##change CLOSED_EYES_FRAME longer for hard blink
                TOTAL_BLINKS += 1
                CEF_COUNTER = 0

        cv.putText(frame, f'Total Blink: {TOTAL_BLINKS}', (10, 30), cv.FONT_HERSHEY_PLAIN, 2, GREEN
                   , 1,
                   cv.LINE_AA)

        (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
        (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)

        r_h_position, r_h_ratio = piece_positioner(center_right, mesh_points[R_H_RIGHT][0],
                                                mesh_points[R_H_LEFT][0], "hor_right")
        l_h_position, l_h_ratio = piece_positioner(center_left, mesh_points[L_H_LEFT][0],
                                                mesh_points[L_H_RIGHT][0], "hor_left")
        h_h_position, h_h_ratio = piece_positioner(mesh_points[NOSE_CENTER][0], mesh_points[234], mesh_points[454], "hor_head")
        v_h_position, v_h_ratio = piece_positioner(mesh_points[NOSE_CENTER][0], mesh_points[10], mesh_points[152], "")
        #print(f'Vertical Head pos: {v_h_position} {v_h_position}')
        #print(f'Horizontall Head pos: {h_h_position} {h_h_position}')

        #print(l_h_position, f"{l_h_ratio:.2f}", r_h_position, f"{r_h_ratio:.2f}")

        cv.putText(frame, f'R: {r_h_position} ', (int(img_w * 0.85), int(img_h)-30), cv.FONT_HERSHEY_PLAIN, 2, RED, 1, cv.LINE_AA)
        cv.putText(frame, f'L: {l_h_position}', (int(img_w * 0.01), int(img_h)-30), cv.FONT_HERSHEY_PLAIN, 2, GREEN, 1, cv.LINE_AA)

        ###TOGGLE HOVER MODE
        if TOTAL_BLINKS == 3 and not fControl :
            fControl = not fControl
            TOTAL_BLINKS = 0
        if TOTAL_BLINKS == 3 and fControl and flight:
            eyemode = not eyemode
            TOTAL_BLINKS = 0


        if fControl:
            cv.putText(frame, f'3 blink during flight to switch Hover Mode', (10, 45), cv.FONT_HERSHEY_PLAIN, 1, GREEN, 1,cv.LINE_AA)
            cv.putText(frame, 'Open mouth or enlarge eyes to change commands', (int(img_w * 0.2) - 50, 120),
                       cv.FONT_HERSHEY_SIMPLEX, 1,
                       GREEN, 1)
        else:
            cv.putText(frame, f'3 long blink to activate Flight Mode', (10, 45), cv.FONT_HERSHEY_PLAIN, 1, GREEN, 1,cv.LINE_AA)



        ###FLIGHT CONTROL SECTION
        if fControl:
            if not flight: TOTAL_BLINKS=0

            if eyemode and flight:
                cv.putText(frame, 'HOVER MODE ACTIVATED', (int(img_w * 0.4) - 50, 30), cv.FONT_HERSHEY_SIMPLEX, 1,
                           CC_COLOR, 3)
            else: cv.putText(frame, 'CONTROL ACTIVATED', (int(img_w * 0.4) - 40, 30), cv.FONT_HERSHEY_SIMPLEX, 1,CC_COLOR, 3)
            #cv.putText(frame, 'CONTROL DEACTIVATED', (int(img_w * 0.4) - 60, 30), cv.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 255), 3)

            ####TRY LAND/TAKEOFF COMMAND
            try:
                # Drone Landing
                if ((h_h_position == "RIGHT" or h_h_position == "LEFT") and eye_open and not eyemode and flight) or key == "l" or key == "L" :
                    counterLand += 1

                    '''
                    if in_circle(int(img_w * 0.2), int(img_h * 0.5), 225, mesh_points[NOSE_CENTER][0]) and not eyemode:
                        cv.putText(frame, f'Land Drone?', mesh_points[346], cv.FONT_HERSHEY_PLAIN, 1.5,
                                   (255, 0, 255), 1, cv.LINE_AA)
                    elif in_circle(int(img_w * 0.8), int(img_h * 0.5), 225,
                                   mesh_points[NOSE_CENTER][0]) and not eyemode:
                        cv.putText(frame, f'Land Drone?', mesh_points[234], cv.FONT_HERSHEY_PLAIN, 1.5,
                                   (255, 0, 255), 1, cv.LINE_AA)
                    '''
                    cv.putText(frame, f'Landing Drone in.. {loading - counterLand}', (int(img_w * 0.38) - 55, 160),
                               cv.FONT_HERSHEY_PLAIN, 2, GREEN,2)
                    if counterLand == loading:
                        if tello_cc:
                            myTello.land()
                            print('Tello Landing!!!!')
                        flight = not flight
                        print(f'flight is {flight}')

                elif (h_h_position == "RIGHT" or h_h_position == "LEFT") and not flight and not eyemode:
                    counterTakeoff += 1
                    if   in_circle(int(img_w * 0.2), int(img_h * 0.5), 225, mesh_points[NOSE_CENTER][0]) and not eyemode:
                        cv.putText(frame, f'Fly Drone?', mesh_points[346], cv.FONT_HERSHEY_PLAIN, 1.5,
                                   (255, 0, 255),2, cv.LINE_AA)
                    elif in_circle(int(img_w * 0.8), int(img_h * 0.5), 225, mesh_points[NOSE_CENTER][0]) and not eyemode:
                        cv.putText(frame, f'Fly Drone?', mesh_points[234], cv.FONT_HERSHEY_PLAIN, 1.5,
                               (255, 0, 255),2, cv.LINE_AA)
                    cv.putText(frame, f'Drone Takeoff in.. {loading - counterTakeoff}', (int(img_w * 0.38) - 55, 160),
                               cv.FONT_HERSHEY_PLAIN, 2, GREEN,2)
                    if counterTakeoff == loading:
                        if tello_cc:
                            myTello.takeoff()
                            print('Tello Takeoff!!!!')
                        flight = not flight
                        print(f'flight is {flight}')
                else:
                    counterTakeoff = 0
                    counterLand = 0

                if (in_circle(int(img_w * 0.2), int(img_h * 0.5), 225, mesh_points[NOSE_CENTER][0]) and flight) and eyemode:
                    cv.putText(frame, f'Landing NOT supported in Hover Mode', (img_w-200, 45), cv.FONT_HERSHEY_PLAIN, 1,
                               RED, 1, cv.LINE_AA)
                elif in_circle(int(img_w * 0.8), int(img_h * 0.5), 225, mesh_points[NOSE_CENTER][0]) and flight and eyemode:
                    cv.putText(frame, f'Takeoff NOT supported in Hover Mode', (img_w-200, 45), cv.FONT_HERSHEY_PLAIN, 1,
                              RED, 1, cv.LINE_AA)
            except:
                pass

            ####TRY DIRECTION/VELOCITY COMMAND
            try:
                if flight and not eyemode:

                    speed = 40
                    lr = 0.0
                    ud = 0.0
                    yv = 0.0
                    fb = 0.0
                    print(f'pass {h_h_position} {v_h_position}')

                    if h_h_position == "LEFT" and not eye_open and not eyemode :
                        cv.putText(frame, f'FACE LEFT', (int(img_w * 0.01), int(img_h * 0.25)),
                                   cv.FONT_HERSHEY_PLAIN, 2, GREEN, 2, cv.LINE_AA)
                        h_h_ratio = -(1-h_h_ratio)
                    if v_h_position == "DOWN":
                        cv.putText(frame, 'FACE DOWN', (int(img_w * 0.5) - 60, 200), cv.FONT_HERSHEY_PLAIN, 2,
                                   RED, 2, cv.LINE_AA)
                        v_h_ratio = -(v_h_ratio)
                    if h_h_position == "RIGHT" and not eye_open and not eyemode:
                        cv.putText(frame, f'FACE RIGHT', (int(img_w * 0.8), int(img_h * 0.25)),
                                   cv.FONT_HERSHEY_PLAIN, 2, RED, 2, cv.LINE_AA)
                    if v_h_position == "UP":
                        cv.putText(frame, 'FACE UP', (int(img_w * 0.5) - 80, 200), cv.FONT_HERSHEY_PLAIN, 2,
                                   GREEN, 2, cv.LINE_AA)
                        v_h_ratio = (1-v_h_ratio)
                    if h_h_position == "CENTER" or counterLand >= 1:
                        h_h_ratio = 0
                    if v_h_position == "MIDDLE" or counterLand >= 1:
                        v_h_ratio = 0

                    print(f'h:{h_h_ratio:.2f} v:{v_h_ratio:.2f} eye_open:{eye_open}')

                    try:
                        if mouth_open:
                            yv = h_h_ratio ##LOOK LEFT/RIGHT
                            ud = v_h_ratio ##GO UP/DOWN
                        else:
                            fb = v_h_ratio #GO FORWARD/BACKWARD
                            lr = h_h_ratio #GO LEFT/RIGHT
                    except:
                        pass

                    # send rc to tello
                    if tello_cc:
                        myTello.send_rc_control(int(lr * speed), int(fb * speed), int(ud * speed), int(yv * speed))
                    print('left idx F: ', (int(lr * speed), int(fb * speed)), 'right idx F: ',
                          (int(ud * speed), int(yv * speed)))

                elif flight and eyemode:

                    if l_h_position == "RIGHT" and r_h_position == "RIGHT":
                        print('LR,FB: (0, 0) UD,YV: (0,30)')
                        cv.putText(frame, f'LOOK RIGHT', (int(img_w * 0.8), int(img_h * 0.25)),
                                   cv.FONT_HERSHEY_PLAIN, 2, RED, 1, cv.LINE_AA)
                        if tello_cc:
                            myTello.send_rc_control(0, 0, 0, 30)
                    elif l_h_position == "LEFT" and r_h_position == "LEFT":
                        cv.putText(frame, f'LOOK LEFT', (int(img_w * 0.01), int(img_h * 0.25)),
                                   cv.FONT_HERSHEY_PLAIN, 2, GREEN, 1, cv.LINE_AA)
                        print('LR,FB: (0, 0) UD,YV: (0,-30)')
                        if tello_cc:
                            myTello.send_rc_control(0, 0, 0, -30)
                    else:
                        print('LR,FB: (0, 0) UD,YV: (0,0)')
                        if tello_cc:
                            myTello.send_rc_control(0, 0, 0, 0)
                ###SAFETY PRECAUTION
                else:
                    if tello_cc:
                        myTello.send_rc_control(0, 0, 0, 0)
                    #cv.circle(frame, (int(img_w * 0.3), int(img_h * 0.45)), 25, CC_COLOR, cv.FILLED)
                    #cv.circle(frame, (int(img_w * 0.7), int(img_h * 0.45)), 25, CC_COLOR, cv.FILLED)
            except:
                #cv.circle(frame, (int(img_w * 0.3), int(img_h * 0.45)), 25, CC_COLOR, cv.FILLED)
                #cv.circle(frame, (int(img_w * 0.7), int(img_h * 0.45)), 25, CC_COLOR, cv.FILLED)
                pass
            '''
            ####OVERLAY TELLO FRAME
            try:
                frame[img_h-240:img_h, img_w-360*2:img_w-360] = telloframe
            except:
                pass
            '''
        else:
            cv.putText(frame, 'CONTROL DEACTIVATED', (int(img_w * 0.4) - 60, 30), cv.FONT_HERSHEY_SIMPLEX, 1,
                       (0,0,255), 3)
            print('LR,FB: (0, 0) UD,YV: (0,0)')
            if tello_cc:
                myTello.send_rc_control(0, 0, 0, 0)
            '''
            cv.putText(frame, 'To activate Tello hand controller', (int(img_w * 0.3) - 45, 300), cv.FONT_HERSHEY_SIMPLEX, 1, CC_COLOR, 2)
            cv.putText(frame, 'move both index fingers', (int(img_w * 0.35) - 40, 340), cv.FONT_HERSHEY_SIMPLEX, 1, CC_COLOR, 2)
            cv.putText(frame, 'into the boxes above', (int(img_w * 0.38) - 43, 380), cv.FONT_HERSHEY_SIMPLEX, 1, CC_COLOR, 2)
            
            try:
                frame[img_h-tf_h:img_h, img_w-tf_w*2:img_w-tf_w] = telloframe
            except:
                pass
            '''

        if flight and not eyemode:
            if mouth_open:
                if eye_open:
                    cv.putText(frame, f'GO LAND', (mesh_points[234] - [100, 0]), cv.FONT_HERSHEY_PLAIN, 1.5,
                               (255, 0, 255), 2, cv.LINE_AA)
                    cv.putText(frame, f'GO LAND', (mesh_points[454] + [0, 0]), cv.FONT_HERSHEY_PLAIN, 1.5,
                               (255, 0, 255), 2, cv.LINE_AA)
                if not eye_open:
                    cv.putText(frame, f'LOOK LEFT', (mesh_points[234] - [120, 0]), cv.FONT_HERSHEY_PLAIN, 1.5,
                               (255, 0, 255), 2, cv.LINE_AA)
                    cv.putText(frame, f'LOOK RIGHT', (mesh_points[454] + [0, 0]), cv.FONT_HERSHEY_PLAIN, 1.5,
                               (255, 0, 255), 2, cv.LINE_AA)
                cv.putText(frame, f'GO DOWN', (mesh_points[152] - [50, -30]), cv.FONT_HERSHEY_PLAIN, 1.5,
                           (255, 0, 255), 2, cv.LINE_AA)
                cv.putText(frame, f'GO UP', (mesh_points[10] - [30, 30]), cv.FONT_HERSHEY_PLAIN, 1.5,
                           (255, 0, 255), 2, cv.LINE_AA)
            if not mouth_open:
                if eye_open:
                    cv.putText(frame, f'GO LAND', (mesh_points[234] - [100, 0]), cv.FONT_HERSHEY_PLAIN, 1.5,
                               (255, 0, 255), 2, cv.LINE_AA)
                    cv.putText(frame, f'GO LAND', (mesh_points[454] + [0, 0]), cv.FONT_HERSHEY_PLAIN, 1.5,
                               (255, 0, 255), 2, cv.LINE_AA)
                if not eye_open:
                    cv.putText(frame, f'GO LEFT', (mesh_points[234] - [100, 0]), cv.FONT_HERSHEY_PLAIN, 1.5,
                               (255, 0, 255), 2, cv.LINE_AA)
                    cv.putText(frame, f'GO RIGHT', (mesh_points[454] + [0, 0]), cv.FONT_HERSHEY_PLAIN, 1.5,
                               (255, 0, 255), 2, cv.LINE_AA)
                cv.putText(frame, f'BACKWARD', (mesh_points[152] - [50, -30]), cv.FONT_HERSHEY_PLAIN, 1.5,
                           (255, 0, 255), 2, cv.LINE_AA)
                cv.putText(frame, f'FORWARD', (mesh_points[10] - [50, 30]), cv.FONT_HERSHEY_PLAIN, 1.5,
                       (255, 0, 255), 2, cv.LINE_AA)

        if not eyemode:
            #cv.circle(frame, mesh_points[NOSE_CENTER][0], 3, GREEN, -1)
            ## lips
            #cv.circle(frame, mesh_points[0], 3, GREEN, -1)
            #cv.circle(frame, mesh_points[17], 3, GREEN, -1)
            ## head points
            cv.circle(frame, mesh_points[10], 3, GREEN, -1) ##up
            cv.circle(frame, mesh_points[454], 3, GREEN, -1) ##right
            cv.circle(frame, mesh_points[234], 3, GREEN, -1) ##left
            cv.circle(frame, mesh_points[152], 3, GREEN, -1) ##down

            #cv.circle(frame, mesh_points[NOSE_CENTER][0], 1, GREEN, 1, cv.LINE_AA)


        #if facemode:
            #landmarksDetection(frame, results, CC_COLOR, 1, True)
            #cv.polylines(frame, [np.array([mesh_points[p] for p in FACE_OVAL], dtype=np.int32)], True, CC_COLOR, 1,
            #            cv.LINE_AA)
        if not flight:
            cv.putText(frame, f'Drone is not flying', (img_w - 180, 45), cv.FONT_HERSHEY_PLAIN, 1, RED, 1, cv.LINE_AA)
            #landmarksDetection(frame, results, CC_COLOR, 1, True)
            # cv.polylines(frame, [np.array([mesh_points[p] for p in FACE_OVAL], dtype=np.int32)], True, CC_COLOR, 1,
            #            cv.LINE_AA)

        #if flight: print("flight")

        if control_center:
            cv.circle(frame, center_left, int(l_radius), GREEN, 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), RED, 1, cv.LINE_AA)

            cv.circle(frame, mesh_points[R_H_RIGHT][0], 2, (255, 255, 255), 1, cv.LINE_AA)
            cv.circle(frame, mesh_points[R_H_LEFT][0], 2, (255, 255, 255), 1, cv.LINE_AA)
            cv.circle(frame, mesh_points[L_H_RIGHT][0], 2, (255, 255, 255), 1, cv.LINE_AA)
            cv.circle(frame, mesh_points[L_H_LEFT][0], 2, (255, 255, 255), 1, cv.LINE_AA)

            cv.polylines(frame, [np.array([mesh_points[p] for p in LEFT_EYE], dtype=np.int32)], True, GREEN
                         , 1,
                         cv.LINE_AA)
            cv.polylines(frame, [np.array([mesh_points[p] for p in RIGHT_EYE], dtype=np.int32)], True, RED, 1,
                         cv.LINE_AA)
            if not eyemode: landmarksDetection(frame, results, CC_COLOR, 1, True)
            cv.imshow("Face & Eye Mode - Control Center", frame)
        else:
            cv.circle(frame, center_left, int(l_radius), GREEN, 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), RED, 1, cv.LINE_AA)
            cv.polylines(frame, [np.array([mesh_points[p] for p in LEFT_EYE], dtype=np.int32)], True, GREEN
                         , 1,
                         cv.LINE_AA)
            cv.polylines(frame, [np.array([mesh_points[p] for p in RIGHT_EYE], dtype=np.int32)], True, RED, 1,
                         cv.LINE_AA)
            cv.imshow("Face & Eye Mode - Webcam ", frame)


        key = cv.waitKey(1)

        ###CC TOGGLE CHECKER
        if key == ord("c") or key == ord("C"):
            control_center = not control_center
            cv.destroyAllWindows()

        ###PROGRAM BREAK CHECKER
        if key == ord("q") or key == ord("Q"):
            mainprogram = not mainprogram
            facemode = not facemode
            ###SAFETY PRECAUTION SECTION
            if flight:
                if tello_cc:
                    myTello.land()
                flight = not flight
                if fControl: fControl = not fControl
            cv.destroyAllWindows()
            break

    cv.imshow("Main Menu", frame)

    key = cv.waitKey(1)

    ###CC TOGGLE CHECKER
    if key == ord("c") or key == ord("C"):
        control_center = not control_center
        cv.destroyAllWindows()

    ###PROGRAM BREAK CHECKER
    if key == ord("e") or key == ord("E"):
        cap.release()
        cv.destroyAllWindows()
        break
    '''
    mp_face_mesh = mp.solutions.face_mesh
    hand_detector = handLandmarkDetector()
    
    CEF_COUNTER = 0
    TOTAL_BLINKS = 0
    CLOSED_EYES_FRAME = 3
    
    cap = cv.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5,
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.flip(frame, 1)
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame = cv.resize(frame, None, fx=1.8, fy=1.8, interpolation=cv.INTER_CUBIC)
            img_h, img_w = frame.shape[:2]
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
    
                mesh_points = np.array(
                    [
                        np.multiply([p.x,p.y],[img_w, img_h]).astype(int)
                        for p in results.multi_face_landmarks[0].landmark
                    ]
                )
    
                # draw control activation circles
                cv.circle(frame, (int(img_w * 0.4), int(img_h * 0.1)), 25, CC_COLOR, 3)
                cv.circle(frame, (int(img_w * 0.6), int(img_h * 0.1)), 25, CC_COLOR, 3)
    
                cv.putText(frame, 'CONTROL ACTIVATED', (int(img_w*0.4)-40, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
                cv.putText(frame, 'CONTROL DEACTIVATED', (int(img_w * 0.4) - 60, 30), cv.FONT_HERSHEY_SIMPLEX, 1,
                           (0, 255, 255), 3)
    
                # left joystick
                cv.circle(frame, (int(img_w * 0.3), int(img_h * 0.45)), 125, CC_COLOR, 15)
                cv.putText(frame, str('Rotation'), (int(img_w * 0.07), int(img_h * 0.475)), cv.FONT_HERSHEY_SIMPLEX, 1,(0,255,255), 2)
                cv.putText(frame, str('Direction'), (int(img_w * 0.245), int(img_h * 0.7)-40), cv.FONT_HERSHEY_SIMPLEX, 1,(0,255,255), 2)
    
                # right joystick
                cv.circle(frame, (int(img_w * 0.7), int(img_h * 0.45)), 125, CC_COLOR, 15)
                cv.putText(frame, str('Yaw'), (int(img_w * 0.83), int(img_h * 0.475)), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                cv.putText(frame, str('Height'), (int(img_w * 0.665), int(img_h * 0.7)-40), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),2)
                # land drone circle
                cv.circle(frame, (int(img_w * 0.4), int(img_h * 0.25)), 25, (0,255,255), 2)
                cv.putText(frame, str('Land'), (int(img_w * 0.4) - 40, int(img_h * 0.25) - 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                # takeoff drone circle
                cv.circle(frame, (int(img_w * 0.6), int(img_h * 0.25)), 25, (0,255,255), 2)
                cv.putText(frame, str('Takeoff'), (int(img_w * 0.6) - 60, int(img_h * 0.25) - 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    
                blink_ratio = blinkRatio(frame, mesh_points, RIGHT_EYE, LEFT_EYE)
                if blink_ratio > 5.5:
                    CEF_COUNTER += 1
                    cv.putText(frame, f'Blink', (240, 30), cv.FONT_HERSHEY_PLAIN, 2, RED
                    , 1, cv.LINE_AA)
    
                else:
                    if CEF_COUNTER > CLOSED_EYES_FRAME:
                        TOTAL_BLINKS += 1
                        CEF_COUNTER = 0
    
                cv.putText(frame, f'Total Blink: {TOTAL_BLINKS}', (10, 30), cv.FONT_HERSHEY_PLAIN, 2, RED
                , 1, cv.LINE_AA)
    
                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                center_left = np.array([l_cx,l_cy],dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)
    
                cv.circle(frame,center_left,int(l_radius),RED
                ,1,cv.LINE_AA)
                cv.circle(frame, center_right, int(r_radius), GREEN, 1, cv.LINE_AA)
    
                cv.circle(frame, mesh_points[R_H_RIGHT][0],2, (255, 255, 255), 1, cv.LINE_AA)
                cv.circle(frame, mesh_points[R_H_LEFT][0], 2, (255, 255, 255), 1, cv.LINE_AA)
                cv.circle(frame, mesh_points[L_H_RIGHT][0], 2, (255, 255, 255), 1, cv.LINE_AA)
                cv.circle(frame, mesh_points[L_H_LEFT][0], 2, (255, 255, 255), 1, cv.LINE_AA)
    
                cv.circle(frame, mesh_points[NOSE_CENTER][0], 4, GREEN, 1, cv.LINE_AA)
    
                cv.polylines(frame, [np.array([mesh_points[p] for p in LEFT_EYE], dtype=np.int32)], True, RED
                , 1,
                             cv.LINE_AA)
                cv.polylines(frame, [np.array([mesh_points[p] for p in RIGHT_EYE], dtype=np.int32)], True, GREEN, 1,
                            cv.LINE_AA)
    
                r_h_position, r_h_ratio = piece_position(center_right,mesh_points[R_H_RIGHT][0],
                                                        mesh_points[R_H_LEFT][0],True)
                l_h_position, l_h_ratio = piece_position(center_left, mesh_points[L_H_LEFT][0],
                                                        mesh_points[L_H_RIGHT][0],False)
    
                print(l_h_position, f"{l_h_ratio:.2f}",r_h_position,f"{r_h_ratio:.2f}")
    
                cv.putText(frame, f'R: {r_h_position} {r_h_ratio:.2f}', (int(img_w *0.8), int(img_h * 0.25)), cv.FONT_HERSHEY_PLAIN, 2, (0,255,0), 1, cv.LINE_AA)
                cv.putText(frame, f'L: {l_h_position} {l_h_ratio:.2f}', (int(img_w *0.01), int(img_h * 0.25)), cv.FONT_HERSHEY_PLAIN, 2, RED
                , 1, cv.LINE_AA)
    
                frame = hand_detector.findHands(frame, draw=True)
                fingerLs = hand_detector.drawFingerPoint(frame)
    
            cv.imshow("img", frame)
            key = cv.waitKey(1)
            if key == ord("q"):
                break
        cap.release()
        cv.destroyAllWindows()
    '''

