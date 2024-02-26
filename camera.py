import cv2
import tensorflow as tf 
import numpy as np

model = tf.keras.models.load_model("../model-3-150.h5")
camera = cv2.VideoCapture(0)
resize = 900
factor = resize/150

if camera.isOpened() == False:
    print("Not opened")

while True:
    ret, frame = camera.read()
    if ret == False:
        print("No return")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = tf.keras.preprocessing.image.img_to_array(frame)
    frame150 = tf.image.resize(frame, (150,150))
    frame900 = tf.image.resize(frame, (resize,resize))
    frame900 = np.array(frame900)
    frame900 = cv2.cvtColor(frame900.astype('uint8'), cv2.COLOR_BGR2RGB) #IMAGE
    frameStanderdized = frame150 / 255
    frameStanderdized = np.array(frameStanderdized)
    frameStanderdized = np.expand_dims(frameStanderdized, axis=0)
    predicted = model.predict(frameStanderdized)[0]
    
    xAnnotationCoordinates = []
    yAnnotationCordinates = []
    for i in range(0,len(predicted),2):
        xAnnotationCoordinates.append(predicted[i]*factor)
        yAnnotationCordinates.append(predicted[i+1]*factor)
    
    for i in range(len(xAnnotationCoordinates)):
        cv2.circle(frame900,(int(xAnnotationCoordinates[i]),int(yAnnotationCordinates[i])), radius=3, color=(0,255,0), thickness=-1)
    cv2.imshow('Video',frame900)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
    

    