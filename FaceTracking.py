#______________Import Librairies_________________#
import cv2
import mediapipe as mp
import time

#_____________Get access to the camera_____________#
cap = cv2.VideoCapture(0)

#____________MediaPipe Processing_____________#
mpFaces = mp.solutions.face_mesh
faceMesh = mpFaces.FaceMesh(max_num_faces=3)

#_________Time________#
current_time = 0
preview_time = 0

mpDraw = mp.solutions.drawing_utils
spec = mpDraw.DrawingSpec(color=(0, 128, 0), thickness=1, circle_radius=2)

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
            results = faceMesh.process(imgRGB)
            # print(results.multi_face_landmarks)

            if results.multi_face_landmarks:
                for face_landmark in results.multi_face_landmarks:
                    print(face_landmark)
                    mpDraw.draw_landmarks(frame, face_landmark, mpFaces.FACEMESH_CONTOURS, spec, spec)

                    for id, lm in enumerate(face_landmark.landmark):
                        # print(lm)
                        image_height, image_width, ic = frame.shape
                        x, y = int(lm.x * image_width), int(lm.y * image_height)
                        print(id, x, y)

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
    cap.release()

    #_______Close all the frames___________#
    cv2.destroyAllWindows()
