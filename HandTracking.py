#______________Import Librairies_________________#
import cv2
import mediapipe as mp
import time

#_____________Get access to the camera_____________#
cap = cv2.VideoCapture(0)

#____________MediaPipe Processing_____________#
mpHands = mp.solutions.hands
hands = mpHands.Hands()

#_________Time________#
current_time = 0
preview_time = 0

mpDraw = mp.solutions.drawing_utils
# frame_per_secondes = 0


#____________Boolean value to check if the program is launch_______________#
isLaunch = True

#__________________If there is not any camera, we'll get an error else the program will be launched
if cap.isOpened() == False:
    print("Error video video cam...")
else:
    while isLaunch:
        success, frame = cap.read()
        
        if success == True:
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            if results.multi_hand_landmarks:
                for hand_landmark in results.multi_hand_landmarks:
                    print(hand_landmark)
                    mpDraw.draw_landmarks(frame, hand_landmark, mpHands.HAND_CONNECTIONS)

            current_time = time.time()
            frame_per_secondes = 1 / (current_time - preview_time)
            preview_time = current_time


            cv2.putText(frame, str(int(frame_per_secondes)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv2.imshow('Test of hand tracking', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                isLaunch = False
        else:
            isLaunch = False

    
    #________Release all ressources_____________#
    cv2.release()

    #_______Close all the frames___________#
    cv2.destroyAllWindows()
