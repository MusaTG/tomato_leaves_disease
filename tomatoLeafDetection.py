# %%
from keras_preprocessing.image import ImageDataGenerator

image_size = 128
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    fill_mode="nearest"
)

train = datagen.flow_from_directory(
    "Tomato_Dataset\\train",
    subset="training",
    target_size=(image_size,image_size),
    batch_size=batch_size,
    class_mode="categorical"
)

val = datagen.flow_from_directory(
    "Tomato_Dataset\\train",
    target_size=(image_size,image_size),
    batch_size=batch_size,
    subset="validation"
)

test_datagen = ImageDataGenerator(rescale=1./255)

test = test_datagen.flow_from_directory(
    "Tomato_Dataset\\val",
    target_size=(image_size,image_size),
    batch_size=batch_size,
    class_mode='categorical'
)

# %%
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout,BatchNormalization

def Model():
    cham_dim=-1
    model = Sequential()
    model.add(Conv2D(128,(3,3),activation="relu",input_shape=(image_size,image_size,3)))
    model.add(MaxPooling2D(3,3))
    model.add(BatchNormalization(axis=cham_dim))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3),activation="relu"))
    model.add(BatchNormalization(axis=cham_dim))
    model.add(Conv2D(128, (3, 3),activation="relu"))
    model.add(BatchNormalization(axis=cham_dim))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3),activation="relu"))
    model.add(BatchNormalization(axis=cham_dim))
    model.add(Conv2D(128, (3, 3),activation="relu"))
    model.add(BatchNormalization(axis=cham_dim))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024,activation="relu"))
    model.add(BatchNormalization(axis=cham_dim))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation="softmax"))

    model.compile(optimizer="RMSprop",loss="categorical_crossentropy",metrics=["acc"])
    model.summary()

    return model
# %%
model = Model()
model.fit_generator(
    train,
    steps_per_epoch=250,
    epochs=8,
    validation_data=val,
    validation_steps=62
)
# %%

# model.save("converttflite.h5")

# %%
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

open("tf_lite_model.tflite","wb").write(tflite_model)

# # %%
# from keras.models import load_model

# model1 = load_model("converttflite.h5")

# pred=model1.predict(test)
# len(pred)

# #%%
# # train.classes
# import pandas as pd
# index = train.class_indices
# print(index)
# # %%
# import numpy as np
# from sklearn.metrics import confusion_matrix,classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns
# def predictor(test_gen,test_step):
#     y_pred = []
#     y_true = test_gen.labels
#     classes = list(test_gen.class_indices.keys())
#     class_count = len(classes)
#     errors = 0
#     preds = model.predict_generator(test_gen,steps=test_step,verbose=1)
#     tests = len(preds)
#     for i, p in enumerate(preds):
#         pred_index = np.argmax(p)
#         true_index = test_gen.labels[i]
#         if pred_index!=true_index:
#             errors+=1
#         y_pred.append(pred_index)
#     acc = (1-errors/tests)*100
#     print(f"there {errors} in {tests} tests f o an accuracy of {acc:6.2f}")
#     ypred = np.array(y_pred)
#     ytrue = np.array(y_true)
#     if class_count<=30:
#         cm = confusion_matrix(ytrue,ypred)
#         plt.figure(figsize=(12,8))
#         sns.heatmap(cm, annot=True, vmin=0, fmt="g")
#         plt.xticks(np.arange(class_count)+.5,rotation=90)
#         plt.yticks(np.arange(class_count)+.5,rotation=90)
#         plt.xlabel("Predicted")
#         plt.ylabel("Actual")
#         plt.title("Confusion Matrix")
#         plt.show()
#     clr = classification_report(y_true, y_pred, target_names=classes, digits= 4) # create classification report
#     print("Classification Report:\n----------------------\n", clr)
#     return errors,tests

# errors,tests = predictor(test,32)
# # %%
# Y_pred = model.predict(test,32)
# y_pred = np.argmax(Y_pred, axis=1)
# len(y_pred)
# # %%
# cm = confusion_matrix(test.classes,y_pred)
# print(cm)
# # %%
# plt.figure(figsize=(12,8))
# sns.heatmap(cm)
# plt.show()
# %%
