# import necessary packages
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import speech_recognition as sr
import pyttsx3
import os



# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
r =sr.Recognizer()

# Load the gesture recognizer model
model = load_model('mp_hand_gesture') #path for the folder

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

# Function to convert text to speech
def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
    engine.setProperty("rate", 140)
    # Use female voice
    engine.setProperty('voice', voice_id)

    
    engine.say(command)
    engine.runAndWait()

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)
    # print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            #print(prediction)
            classID = np.argmax(prediction)
            #print(classID)
            className = classNames[classID]
            if(classID==2):
                SpeakText("access granted")
                SpeakText("Hello Good evening everyone !")
                SpeakText("Hi!My name is invi")
                SpeakText("Welcome to our beloved principal sir and all the dignitaries and students present over here.")
                SpeakText("Thank you to all the staff for always encouraging and being supportive to students")
                SpeakText("Today we are celebrating poster launch event from every department")
                SpeakText("congratulations to one and all for making this event grant success")
                SpeakText("This year innovative thinkers from information technology came up with special idea of clapping sensor.")
                SpeakText(" We request our principal sir Dr. J. Sudhakar rao gaaru to initiate the poster launch of IT department")
                SpeakText('thank you everyone for your valuable time')
                exit()
            else:
                print("try again")
                    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break


# release the webcam and destroy all active windows

cap.release()

cv2.destroyAllWindows()
