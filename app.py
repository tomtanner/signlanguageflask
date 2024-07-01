from flask import Flask, render_template,request,Response
import time
import cv2
import os
import json
lst=os.listdir("static")
lst=set(lst)
import subprocess
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import queue
import threading

import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 150)    # Speed of speech
engine.setProperty('volume', 0.9) 


engine.runAndWait()
# Function to speak text
def speak(text):
    engine.say(text)

speech_queue = queue.Queue()

def process_speech_requests():
    while True:
        if not speech_queue.empty():
            text_to_speak = speech_queue.get()
            threading.Thread(target=speak, args=(text_to_speak,)).start()

cap = None  # Global variable to hold the webcam capture object
detector = HandDetector(maxHands=5)
classifier = Classifier("./Model\keras_model.h5" , ".\Model\labels.txt")
offset = 20
imgSize = 300
labels = ["Hello","I love you","No","Okay","Please","Thank you","Yes","Bye","Sorry"]



app = Flask(__name__, static_url_path='/static')

@app.route('/')
def main():
    return render_template('main.html')
    # return render_template('index.html', folder_name=None, video_file_name=None)



@app.route('/index')
def index():
    return render_template('index.html')


def generate_frames():
    global cap
    while True:
     try:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

            imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
                prediction , index = classifier.getPrediction(imgWhite, draw= False)
                current_word = labels[index]
                speak(current_word)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize
                prediction , index = classifier.getPrediction(imgWhite, draw= False)
                
        
            cv2.rectangle(imgOutput,(x-offset,y-offset-70),(x -offset+400, y - offset+60-50),(0,255,0),cv2.FILLED)  

            cv2.putText(imgOutput,labels[index],(x,y-30),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),2) 
            cv2.rectangle(imgOutput,(x-offset,y-offset),(x + w + offset, y+h + offset),(0,255,0),4)   

            # cv2.imshow('ImageCrop', imgCrop)
            # cv2.imshow('ImageWhite', imgWhite)

        cv2.imshow('Image', imgOutput)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
     except:
        pass

@app.route('/video_feed')
def video_feed():
    import cv2
    global cap
    # cap = cv2.VideoCapture(0)
    # threading.Thread(target=generate_frames).start()
    # threading.Thread(target=process_speech_requests).start()
    subprocess.Popen(["python3","static/test.py"])
    #return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Redirect To Sign Language Detector"

@app.route('/play_video', methods=['POST'])
def play_video():
    video_list=[]
    folder_names = request.form.get('folder_names')
    flist=folder_names.split(" ")
    for folder_name in flist:
     if folder_name not  in lst:
        continue
     video_file_path = get_video_path(folder_name)
     if video_file_path:
        directory = os.path.join(app.root_path, 'static', folder_name)
        video_files = os.listdir(directory)
        if video_files:
            # Get the first video file in the folder
            video_file_name = next((f for f in video_files if f.endswith('.mp4')), None)
            if video_file_name:
                 video_list.append([folder_name,video_file_name])
    return render_template('index.html',video_list=video_list,length=len(video_list))

def get_video_path(folder_name):
    if folder_name is None:
        return None

    static_path = os.path.join(app.root_path, 'static')
    print("Static directory path:", static_path)
    folder_path = os.path.join(static_path, folder_name)
    print("Folder path:", folder_path)
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
        if video_files:
            return os.path.join(folder_name, video_files[0])
    return None

if __name__ == '__main__':
    app.run(debug=True)
