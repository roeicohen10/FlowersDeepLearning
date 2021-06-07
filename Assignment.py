import tensorflow
import pandas as pd
import keras
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import scipy.io
import os
import shutil
from sklearn.model_selection import train_test_split


def pre_process(matPath,path,dirName):
    mat = scipy.io.loadmat(matPath)
    for image in mat["labels"][0]:
        if not os.path.exists(dirName+'\\'+str(image)):
            os.mkdir(dirName+'\\'+str(image))
    files = os.listdir(path)
    for file in files:
        file_index  = int(file.split("_")[1].split(".")[0])
        shutil.copy(path+'\\'+file, dirName+'\\'+str(mat["labels"][0,file_index-1]))


def read_images(dir_path):
    df_list = []
    images_arr = []
    for label in os.listdir(dir_path):
        for img in os.listdir(dir_path+'\\'+label):
            image_details = [dir_path+'\\'+label+'\\'+img,label]
            df_list.append(image_details)
    path_df = pd.DataFrame(df_list,columns=['path','label'])
    for index,img in path_df.iterrows():
        image_path = img["path"]
        image = Image.open(image_path)
        image.load()
        image = image.resize((200,200),Image.ANTIALIAS)
        pixels = np.asarray(image,dtype="int32")
        images_arr.append([pixels,int(img["label"])-1])

    images_df = pd.DataFrame(images_arr,columns=["image","label"])
    return images_df


def run_model(model_name,v_size,t_size,df):
    X_train,y_train,X_validation,y_validation,X_test,y_test = split_images_data(v_size,t_size,df)

    datagen = ImageDataGenerator()
    datagen.fit(np.array([x for x in X_train]))

    model = get_model_by_name(model_name)

    callback = callbacks()
#     X_train= np.asarray(X_train).astype(np.float32)

    model_fitted = model.fit(np.array([x for x in X_train]),
        y_train,
        batch_size=250,
        validation_data=(np.array([x for x in X_validation]), y_validation),
        epochs=25,
        shuffle=True,
        callbacks=[callback]
    )

    evaluation = model.evaluate(np.array([x for x in X_test]), y_test)
    cross_entropy_plot(model_fitted)
    accuracy_plot(model_fitted)

    print(model.metrics_names[0] + ":" + str(evaluation[0]))
    print(model.metrics_names[1] + ":" + str(evaluation[1]))


def cross_entropy_plot(model_fitted):
    plt.plot(model_fitted.history['loss'])
    plt.plot(model_fitted.history['val_loss'])
    plt.title('Cross Entropy')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()


def accuracy_plot(model_fitted):
    plt.plot(model_fitted.history['accuracy'])
    plt.plot(model_fitted.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()


def callbacks():
    return [keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)]


def get_model_by_name(model_name):
    model = Sequential()
    if model_name=="VGG":
        keras_model = VGG16(weights='imagenet', include_top=False, input_shape=(200,200,3))
    else:
        keras_model = ResNet50(weights='imagenet', include_top=False, input_shape=(200,200,3))
    for layer in keras_model.layers:
        layer.trainable = False
    model.add(keras_model)
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(500, activation='selu'))
    model.add(keras.layers.Dropout(.4))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(256, activation='selu'))
    model.add(keras.layers.Dropout(.25))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(102, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    return model


def label_to_vec(data,labels_num):
    vectors=[]
    data = np.array(data)
    data = data.reshape(-1, 1)
    labels = np.array(data)
    labels = labels.reshape(-1)
    matrix_label=np.eye(labels_num)
    for l in labels:
        vectors.append(matrix_label[int(l)])
    return vectors

def split_images_data(validation_size,test_size,df):

    X_train, X_test, y_train, y_test=train_test_split(df["image"],df["label"],test_size=test_size,stratify=df['label'])
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    y_train,y_test,y_validation = vectors(y_train,y_test,y_validation)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_validation = np.array(X_validation)
    y_validation = np.array(y_validation)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train,y_train,X_validation,y_validation,X_test,y_test


def vectors(y_train,y_test,y_validation):
    y_train = label_to_vec(y_train,102)
    y_test = label_to_vec(y_test,102)
    y_validation = label_to_vec(y_validation,102)
    return y_train,y_test,y_validation


if __name__ == '__main__':
    images_path = r'pre/jpg'
    dstPath = r'data'
    mat_path = r'pre/imagelabels.mat'
#     pre_process(mat_path,images_path,dstPath)
    df = read_images(dstPath)
    run_model("VGG",0.3,0.4,df)
    # run_model("ResNet",0.3,0.4,df)