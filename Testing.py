from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
from playsound import playsound
import threading
import time


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model('C:/Users/Kevin Vo/Desktop/ForAIClass/keras_model.h5', compile=False)

# Load the labels
class_names = open('C:/Users/Kevin Vo/Desktop/ForAIClass/labels.txt', "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(1)

def chair():
    threading.Thread(target=playsound, args=('C:/Users/Kevin Vo/Desktop/ForAIClass/chair.mp3',), daemon=True).start()

def door():
    threading.Thread(target=playsound, args=('C:/Users/Kevin Vo/Desktop/ForAIClass/door.mp3',), daemon=True).start()


def crosswalk():
    threading.Thread(target=playsound, args=('C:/Users/Kevin Vo/Desktop/ForAIClass/road.mp3',), daemon=True).start()

def stair():
    threading.Thread(target=playsound, args=('C:/Users/Kevin Vo/Desktop/ForAIClass/stairs.mp3',), daemon=True).start()

def table():
    threading.Thread(target=playsound, args=('C:/Users/Kevin Vo/Desktop/ForAIClass/table.mp3',), daemon=True).start()

def person():
    threading.Thread(target=playsound, args=('C:/Users/Kevin Vo/Desktop/ForAIClass/person.mp3',), daemon=True).start()


while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
  

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # if(class_names[1] == class_name and 90 < np.round(confidence_score * 100)):
    #     door()
    # elif(class_names[2] == class_name and 90 < np.round(confidence_score * 100)):  
    #     crosswalk()
    # elif(class_names[3] == class_name and 90 < np.round(confidence_score * 100)):
    #     chair()
    # elif(class_names[5] == class_name and 90 < np.round(confidence_score * 100)):
    #     table()
    # elif(class_names[7] == class_name and 90 < np.round(confidence_score * 100)):
    #     person()
    # elif(class_names[0] == class_name and 90 < np.round(confidence_score * 100)):
    #     stair()

    if(80 < np.round(confidence_score * 100)):
        if(class_names[1] == class_name):
            door()
        elif(class_names[2] == class_name):  
            crosswalk()
        elif(class_names[3] == class_name):
            chair()
        elif(class_names[5] == class_name):
            table()
        elif(class_names[7] == class_name):
            person()
            print("obama")
        elif(class_names[0] == class_name ):
            stair()

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
