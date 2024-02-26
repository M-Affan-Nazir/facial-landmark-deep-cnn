import tensorflow.compat.v1 as tf
from dataGenerator import DataGenerator


def modelArchitecture1(imageDim):
    cnn = tf.keras.models.Sequential()

    cnn.add(tf.keras.layers.Conv2D(filters = 30,kernel_size=(3,3),activation="relu", input_shape=[imageDim,imageDim,3]))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    cnn.add(tf.keras.layers.BatchNormalization())

    cnn.add(tf.keras.layers.Conv2D(filters=30,kernel_size=(3,3),strides=(1,1), activation="relu"))
    cnn.add(tf.keras.layers.BatchNormalization())
    cnn.add(tf.keras.layers.Conv2D(filters=30,kernel_size=(3,3),strides=(1,1), activation="relu"))
    cnn.add(tf.keras.layers.BatchNormalization())
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    cnn.add(tf.keras.layers.Conv2D(filters=30,kernel_size=(3,3),strides=(1,1), activation = "relu"))
    cnn.add(tf.keras.layers.BatchNormalization())
    cnn.add(tf.keras.layers.Conv2D(filters=30,kernel_size=(3,3),strides=(1,1), activation="relu"))
    cnn.add(tf.keras.layers.BatchNormalization())
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    cnn.add(tf.keras.layers.Conv2D(filters=30,kernel_size=(3,3),strides=(1,1), activation = "relu"))
    cnn.add(tf.keras.layers.BatchNormalization())
    cnn.add(tf.keras.layers.Conv2D(filters=30,kernel_size=(3,3),strides=(1,1), activation="relu"))
    cnn.add(tf.keras.layers.BatchNormalization())
    # cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    cnn.add(tf.keras.layers.Conv2D(filters=50,kernel_size=(3,3),strides=(1,1), activation = "relu")) #128
    cnn.add(tf.keras.layers.BatchNormalization())
    cnn.add(tf.keras.layers.Conv2D(filters=50,kernel_size=(3,3),strides=(1,1), activation = "relu")) #128
    cnn.add(tf.keras.layers.BatchNormalization())
    # cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    cnn.add(tf.keras.layers.Conv2D(filters=100, kernel_size=(3,3),strides=(1,1), activation="relu")) #256

    cnn.add(tf.keras.layers.Flatten())

    cnn.add(tf.keras.layers.Reshape((5,500)))
    cnn.add(tf.keras.layers.LSTM(units = 701, activation = "relu"))
    cnn.add(tf.keras.layers.Dense(units = 501, activation = "relu"))
    cnn.add(tf.keras.layers.Dense(units = 501, activation = "relu"))
    cnn.add(tf.keras.layers.Dense(units = 501, activation = "relu"))
    
    cnn.add(tf.keras.layers.Dropout(0.2))
    
    cnn.add(tf.keras.layers.Dense(units=388))


    cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss = "mse")


    print("Model Compiled Successfully...")

    return cnn

def trainModelUsingDataGenerator(archetectureNum,model, imageDim):
    print("Training using Archetecture Number: " + str(archetectureNum))
    checkPoint = tf.keras.callbacks.ModelCheckpoint(
                                                        "../modelTrainingCheckpoint/model-"+str(archetectureNum)+"-"+str(imageDim)+".h5",
                                                        # "./modelTrainingCheckpoint/model-"+str(imageDim),
                                                        monitor = "val_loss",
                                                        save_best_only=True,
                                                        mode='min',
                                                        save_freq='epoch'
                                                    )
    
    TrainingGenerator = DataGenerator("../helen_dataset/train/", "../helen_dataset/scaledAnnotation-"+str(imageDim)+"/trainAnnotation/", batch_size=32, dim=(imageDim,imageDim))
    TestingGenerator = DataGenerator("../helen_dataset/test/",   "../helen_dataset/scaledAnnotation-"+str(imageDim)+"/testAnnotation/" , batch_size=32, dim=(imageDim,imageDim))
    cnn = model
    cnn.fit(TrainingGenerator, validation_data = TestingGenerator, epochs = 500, steps_per_epoch = len(TrainingGenerator), validation_steps = len(TestingGenerator), callbacks=[checkPoint])
    print()
    print("Training Complete")

def continueTraining(file, additionalEpochs, imageDim):
    checkPoint = tf.keras.callbacks.ModelCheckpoint(
                                                        file,
                                                        monitor = "val_loss",
                                                        save_best_only=True,
                                                        mode='min',
                                                        save_freq='epoch'
                                                    )
    
    TrainingGenerator = DataGenerator("../helen_dataset/train/", "../helen_dataset/scaledAnnotation-"+str(imageDim)+"/trainAnnotation/", batch_size=32, dim=(imageDim,imageDim))
    TestingGenerator = DataGenerator("../helen_dataset/test/",   "../helen_dataset/scaledAnnotation-"+str(imageDim)+"/testAnnotation/" , batch_size=32, dim=(imageDim,imageDim))
    cnn = tf.keras.models.load_model(file)
    cnn.fit(TrainingGenerator, validation_data = TestingGenerator, epochs = additionalEpochs, steps_per_epoch = len(TrainingGenerator), validation_steps = len(TestingGenerator), callbacks=[checkPoint])
    print()
    print("Training Complete from Continuation Procedure")
    return cnn


continueTrain = False
imageDim = 150
archetectureNum = 3 #Being used as Variant

if continueTrain == True:
    additionalEpochs = 100
    trainedCnn = continueTraining("../modelTrainingCheckpoint/checkPointModelArchetecture-"+str(archetectureNum)+".h5", additionalEpochs, imageDim)
    trainedCnn.save("../helenTrainedNeuralNetwork")
else:
    cnnModel = modelArchitecture1(imageDim)
    trainModelUsingDataGenerator(archetectureNum, cnnModel,imageDim)
    cnnModel.save("../helenTrainedNeuralNetwork")