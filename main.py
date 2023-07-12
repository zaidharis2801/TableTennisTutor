import cv2
import os
import PySimpleGUI as sg
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import threading
import random
model_final = load_model('model.h5')

# Load your trained model
data_dir=r"E:\classificationTask\train"
label_names = os.listdir(data_dir)
label_encoder = LabelEncoder()
label_encoder.fit(label_names)

# Function to process the video and make predictions
# Function to process the video and make predictions
def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (100, 100))
        frames.append(frame)
    cap.release()
    video = np.array(frames)

    # Preprocess the video data
    input_shape = model.input_shape[1:]
    # padded_video = np.zeros(input_shape)
    n_frames = video.shape[0]
    # Determine the amount of padding needed
    padding_needed = np.maximum(0, input_shape[0] - n_frames)

    # Pad the video frames with zeroes
    padded_video = np.pad(video, [(0, padding_needed), (0, 0), (0, 0), (0, 0)], mode='constant')

    # Normalize the padded video
    padded_video = padded_video.astype('float32') / 255.0

    shots = []
    prediction = model.predict(np.expand_dims(padded_video, axis=0))
    pred = label_encoder.inverse_transform([np.argmax(prediction)])

    return pred

# Function to play the video in a separate thread
def play_video(video_path,title):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        cv2.imshow(title, frame)
        cv2.waitKey(25)
    cv2.destroyAllWindows()


# Define the GUI layout
layout = [
    [sg.Text("Select a video file:")],
    [sg.Input(key="video_path"), sg.FileBrowse()],
    [sg.Button("Analyze"), sg.Button("Exit")],
    [sg.Output(size=(80, 20), key="output")],
]

# Create the window
window = sg.Window("Table Tennis Shot Classifier", layout)

# Event loop
while True:
    event, values = window.read()

    if event in (sg.WIN_CLOSED, "Exit"):
        break

    if event == "Analyze":
        video_path = values["video_path"]
        if video_path:
            t = threading.Thread(target=play_video, args=(video_path,"Video to be analysed",))
            t.start()
            shots = process_video(video_path, model_final)
            print(shots)
            window["output"].update("\n".join(shots))

            folder_path=os.path.join(data_dir,str(shots[0]))
            #print(folder_path)
            video_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]

            #Choose a random video file from the list
            video_path2 = os.path.join(folder_path, random.choice(video_files))
            # print(video_path2)
            t = threading.Thread(target=play_video, args=(video_path2, "Expert Video",))
            t.start()
            # play_video(video_path2,"Expert Video")
        else:
            sg.popup("Please select a video file")

# Close the window
window.close()
