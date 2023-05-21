from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import os
import cv2
import paho.mqtt.client as paho
import sys


client = paho.Client()
message_received = False
if client.connect("192.168.247.205", 1883, 60) != 0:
    print("Could not connect to MQTT Broker!")
    sys.exit(-1)

print("Connected to the MQTT Broker")

def on_message(client, userdata, message):
    print("Message received")
    global message_received
    message_received = True


def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("esp8266/commandToModel")

def dpm(frame, face_net, mask_net):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.8:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            (start_x, start_y) = (max(0, start_x), max(0, start_y))
            (end_x, end_y) = (min(w - 1, end_x), min(h - 1, end_y))
            face = frame[start_y:end_y, start_x:end_x]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((start_x, start_y, end_x, end_y))
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = mask_net.predict(faces, batch_size=32)
    return (locs, preds)

prototxt_path = r"face_detector\deploy.prototxt"
weights_path = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNet(prototxt_path, weights_path)

mask_net = load_model("mask_detector.model")

print("[INFO] starting video stream...")
video_stream = VideoStream(src=0).start()

client.on_connect = on_connect
client.on_message = on_message

client.loop_start()
while True:
    frame = video_stream.read()
    frame = imutils.resize(frame, width=800)
    (locs, preds) = detect_and_predict_mask(frame, face_net, mask_net)

    label = ""
    for (box, pred) in zip(locs, preds):
        (start_x, start_y, end_x, end_y) = box
        (mask, without_mask) = pred

        if mask > without_mask:
            label = "Mask"
            if message_received:
                message_received = False
                print("Opening gate")
                client.publish("python/commandToGate", "Mask", 0)
            else:
                print("Please wait for the previous person to complete the process")
        else:
            label = "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, without_mask) * 100)

        cv2.putText(frame, label, (start_x, start_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        client.loop_stop()
        break
cv2.destroyAllWindows()
video_stream.stop()
