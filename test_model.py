import tensorflow.compat.v1 as tf
import os
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

testAnnotation = os.listdir("../helen_dataset/orignalAnnotations/testAnnotation")
testingImages = os.listdir("../helen_dataset/test")

archetectureNum = 3 

def predictionUsingImageTransformation(modelpredSize, reSize, imagePath = None, modelPath = None):

    if modelPath != None:
        print("Using Custom Model Path: " + modelPath)
        modelx = tf.keras.models.load_model(modelPath)
    else:
        modelx = tf.keras.models.load_model("../modelTrainingCheckpoint/model-"+str(archetectureNum)+"-"+str(modelpredSize)+".h5",)
    
    print(modelx.summary())
    
    factor = reSize/modelpredSize
    print("model : " + str(modelpredSize))
    print("resized to : " + str(reSize))
    print("-----------------------------------------")

    if imagePath == None:
        for i in range(len(testAnnotation)):
            with open("../helen_dataset/scaledAnnotation-100/testAnnotation/"+testAnnotation[i]) as f:
                imageName  = ""
                xAnnotationCoordinates = []
                yAnnotationCordinates = []

                imageName = f.readline()
                imageName = imageName.strip() + ".jpg"
                
                img = image.load_img("../helen_dataset/test/"+imageName, target_size=(modelpredSize,modelpredSize))
                imageArray = image.img_to_array(img)
                imageArray = imageArray / 255
                imageArray = np.array(imageArray)
                imageArray = np.expand_dims(imageArray, axis=0)

                predicted = modelx.predict(imageArray)[0]
                for i in range(0,len(predicted),2):
                    xAnnotationCoordinates.append(predicted[i]*factor)
                    yAnnotationCordinates.append(predicted[i+1]*factor)

                img = image.load_img("../helen_dataset/test/"+imageName, target_size=(reSize,reSize))
                

                plt.imshow(img)
                plt.title(str(modelpredSize) + " --> " + str(reSize))


                for i in range(len(xAnnotationCoordinates)):
                    plt.scatter(x=xAnnotationCoordinates[i],y=yAnnotationCordinates[i],c="red",s=1)
                plt.show()
    else:
        xAnnotationCoordinates = []
        yAnnotationCordinates = []

        img = image.load_img(imagePath, target_size=(modelpredSize,modelpredSize))
        imageArray = image.img_to_array(img)
        imageArray = imageArray / 255
        imageArray = np.array(imageArray)
        imageArray = np.expand_dims(imageArray, axis=0)

        predicted = modelx.predict(imageArray)[0]
        for i in range(0,len(predicted),2):
            xAnnotationCoordinates.append(predicted[i]*factor)
            yAnnotationCordinates.append(predicted[i+1]*factor)
        
        img = image.load_img(imagePath, target_size=(reSize,reSize))
        plt.imshow(img)
        plt.title(str(modelpredSize) + " --> " + str(reSize))
        for i in range(len(xAnnotationCoordinates)):
                    plt.scatter(x=xAnnotationCoordinates[i],y=yAnnotationCordinates[i],c="red",s=1)
        plt.show()
        predictionUsingImageTransformation(modelpredSize, reSize, modelPath="../model-3-150.h5")


predictionUsingImageTransformation(150,1100, "C:/users/maffa/desktop/photo3.jpg", "../model-3-150.h5")