# %%
from keras_preprocessing.image import ImageDataGenerator

image_size = 224
batch_size = 8

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
    subset="validation",
    class_mode="categorical"
)

test_datagen = ImageDataGenerator(rescale=1./255)

test = test_datagen.flow_from_directory(
    "Tomato_Dataset\\val",
    target_size=(image_size,image_size),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical'
)


# %%
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout,BatchNormalization

def Model():
    cham_dim=-1
    model = Sequential()
    model.add(Conv2D(128,(3,3),activation="relu",input_shape=(image_size,image_size,3),padding="same"))
    model.add(MaxPooling2D(3,3))
    model.add(BatchNormalization(axis=cham_dim))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3),activation="relu",padding="same"))
    model.add(BatchNormalization(axis=cham_dim))
    model.add(Conv2D(128, (3, 3),activation="relu",padding="same"))
    model.add(BatchNormalization(axis=cham_dim))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3),activation="relu",padding="same"))
    model.add(BatchNormalization(axis=cham_dim))
    model.add(Conv2D(128, (3, 3),activation="relu",padding="same"))
    model.add(BatchNormalization(axis=cham_dim))
    model.add(MaxPooling2D(3,3))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1024,activation="relu"))
    model.add(BatchNormalization(axis=cham_dim))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation="softmax"))

    model.summary()

    return model
# %%
model12 = Model()
model12.compile(optimizer="RMSProp",loss="categorical_crossentropy",metrics=["acc"])
history12 = model12.fit_generator(
    train,
    epochs=12,
    validation_data=val
)

# %%
model50 = Model()
model50.compile(optimizer="RMSProp",loss="categorical_crossentropy",metrics=["acc"])
history50 = model50.fit_generator(
    train,
    steps_per_epoch=2000,
    epochs=50,
    validation_data=val,
    validation_steps=500
)


# %%
# from keras.models import load_model

# model1 = load_model("converttflite.h5")

# pred=model1.predict(test)
# len(pred)

# #%%
# # train.classes
# import pandas as pd
# index = train.class_indices
# print(index)
# %%
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
def predictor(test_gen,test_step):
    y_pred = []
    y_true = test_gen.labels
    classes = list(test_gen.class_indices.keys())
    class_count = len(classes)
    errors = 0
    preds = model12.predict_generator(test_gen,steps=test_step,verbose=1)
    tests = len(preds)
    for i, p in enumerate(preds):
        pred_index = np.argmax(p)
        true_index = test_gen.labels[i]
        if pred_index!=true_index:
            errors+=1
        y_pred.append(pred_index)
    acc = (1-errors/tests)*100
    print(f"there {errors} in {tests} tests f o an accuracy of {acc:6.2f}")
    ypred = np.array(y_pred)
    ytrue = np.array(y_true)
    if class_count<=30:
        cm = confusion_matrix(ytrue,ypred)
        plt.figure(figsize=(12,8))
        sns.heatmap(cm,cmap="Blues", annot=True, vmin=0, fmt="g")
        plt.xticks(np.arange(class_count)+.5,rotation=90)
        plt.yticks(np.arange(class_count)+.5,rotation=90)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
    clr = classification_report(y_true, y_pred, target_names=classes, digits= 4) # create classification report
    print("Classification Report:\n----------------------\n", clr)
    return errors,tests

errors,tests = predictor(test,250)

#%%
import numpy as np
from sklearn.metrics import confusion_matrix
Y_pred = model12.predict_generator(test,250)
y_pred = np.argmax(Y_pred,axis=1)
len(y_pred) 
# %% cm
ytrue=np.array(np.array(test.labels))
ypred=np.array(y_pred)
cm=confusion_matrix(ytrue,ypred)
# %%
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,8))
sns.heatmap(cm,cmap="Blues", annot=True)
plt.show()
#%%
from sklearn.metrics import classification_report
clr = classification_report(ytrue, ypred, target_names=list(test.class_indices.keys()), digits= 4) # create classification report
print("Classification Report:\n----------------------\n", clr)
# %% image predict
my_image = load_img('Tomato_Dataset\\val\\Tomato___Tomato_mosaic_virus\\0dae2780-43e7-40ac-ae45-95e5318c8f32___PSU_CG 2290.jpg', target_size=(128,128))
my_image = img_to_array(my_image)
my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
my_image = preprocess_input(my_image)
prediction = model12.predict(my_image)
prediction

# %%

# model.save("converttflite.h5")

# %%
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model12)

tflite_model = converter.convert()

open("tf_lite_model.tflite","wb").write(tflite_model)

# %% control

interpreter = tf.lite.Interpreter(model_path="tf_lite_model.tflite")
input_details=interpreter.get_input_details()
output_details=interpreter.get_output_details()

print("Input shape:",input_details[0]["shape"])
print("Input type:",input_details[0]["dtype"])
print("Output shape:",output_details[0]["shape"])
print("Output type:",output_details[0]["dtype"])

# %%
import matplotlib.pyplot as plt
acc1_original=model12.history["acc"]
val_acc1_original=model12.history["val_acc"]
loss1_original=model12.history["loss"]
val_loss1_original=model12.history["val_loss"]

epochs=range(1,len(acc1_original)+1)
val_epochs=range(1,len(val_acc1_original)+1)


sizeplt=plt.figure()
sizeplt.set_figwidth(16)
sizeplt.set_figheight(6)

plt.plot(epochs,acc1_original,'bo',label="Eğitim başarımı(original)")
plt.plot(val_epochs,val_acc1_original,'b--',label="Doğrulama başarımı(original)")
plt.title("Eğitim ve Doğrulama başarımı 12")
plt.legend()
plt.figure()


sizeplt=plt.figure()
sizeplt.set_figwidth(16)
sizeplt.set_figheight(6)

plt.plot(epochs,loss1_original,'ro',label="Eğitim kaybı(original)")
plt.plot(val_epochs,val_loss1_original,'r--',label="Doğrulama kaybı(original)")
plt.title("Eğitim ve Doğrulama kaybı 12")
plt.legend()

plt.show()
# %%
