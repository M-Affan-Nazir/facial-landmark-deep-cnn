import os
import tensorflow as tf
#from keras.preprocessing import image
from tensorflow.compat.v1.keras.preprocessing import image
import numpy as np



class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, imageDirectory, annotationDirectory, batch_size=32,dim=(300,300),shuffle=True):
        #initialize setup parameters
        self.imageDir = imageDirectory
        self.annoDir = annotationDirectory
        self.batch_size = batch_size
        self.imageDimension = dim
        self.shuffle = shuffle
        self.AnnotationList = os.listdir(self.annoDir)
        print("Dimension: " + str(self.imageDimension))
        print("Batch Size: " + str(self.batch_size))
        pass


    def __len__(self):
        #returns how many batches are to be produced. Model uses this to know how many times it needs to call __get_item__ (__get_item__ returns one batch of data)
        length = len(self.AnnotationList)
        return int(np.floor(length / self.batch_size))


    def __getitem__(self,batchNumber):
        #retrieves one batch of data; called automatically by model when it seeks a new batch (models expects preprocessed; ready data that I can train on)
        #logic to simply retrieve a "batch_size" amount of data, that is the only purpose. Purely raw. Processing the data to be done in __data_generation()
        #returns x,y (x = input; y = expected output).   [x,y taken from __data_generation.]
        start = self.batch_size*batchNumber
        end = start + self.batch_size - 1
        annotationsBatch = []
        
        for i in range(start,end+1):
            annotationsBatch.append(self.AnnotationList[i])
        
        x,y = self.__data_generation(annotationsBatch)
        
        return x,y

    def on_epoch_end(self):
        #methods run at the end of each epoch
        #mainly to shuffle data; so model sees data in different order next time.
        return

    def __data_generation(self,listOfItems):
        #core method; this is where you process rawly loaded data (from __getitem__).
        #this function is called by __getitem__. listOfItems = list of all loaded items.
        #return x,y to __getitem__. [x= processed input (understandable by model); y = processed output (understandable by model)]
        x = []
        y = []
        
        for item in listOfItems:
            with open(self.annoDir + item) as f:
                imageName = f.readline()
                imageName = imageName.strip() + ".jpg"
                img = image.load_img(self.imageDir + imageName, target_size=self.imageDimension)
                imageArray = image.img_to_array(img)
                imageArray = imageArray / 255
                x.append(imageArray)

                tempCordArray = []

                data = f.readlines()
                for j in  range(len(data)):
                    coordinates = data[j].split(",")
                    xCoord = float(coordinates[0].strip())
                    yCoord = float(coordinates[1].strip())
                    tempCordArray.append(xCoord)
                    tempCordArray.append(yCoord)
                
                y.append(tempCordArray)
        
        x = np.array(x)
        y = np.array(y)

        return x,y