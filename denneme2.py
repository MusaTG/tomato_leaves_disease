#%%
from keras_preprocessing.image import ImageDataGenerator

image_size = 128
batch_size = 4

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train = datagen.flow_from_directory(
    "Tomato_Dataset\\train",
    subset="training",
    seed=123,
    target_size=(image_size,image_size),
    batch_size=batch_size,
    class_mode="categorical"
)

val = datagen.flow_from_directory(
    "Tomato_Dataset\\train",
    target_size=(image_size,image_size),
    seed=123,
    batch_size=batch_size,
    subset="validation",
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test = test_datagen.flow_from_directory(
    "Tomato_Dataset\\val",
    target_size=(image_size,image_size),
    batch_size=batch_size,
    class_mode='categorical'
)
# %%
from keras.layers import Conv2D,MaxPooling2D,MaxPool2D,Flatten,Dense,Dropout,BatchNormalization
from keras.models import Sequential
# model = Sequential()
# model.add(Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(image_size,image_size,3)))
# model.add(MaxPooling2D((2,2)))
# model.add(Conv2D(64,(3,3),activation="relu"))
# model.add(MaxPooling2D((2,2)))
# model.add(Conv2D(64,(3,3),activation="relu"))
# model.add(Flatten())
# model.add(Dense(64,activation="relu"))
# model.add(Dense(10,activation="softmax"))
# model = Sequential()
# model.add(Conv2D(128,(3,3),activation="relu",input_shape=(image_size,image_size,3),padding="same"))
# model.add(MaxPooling2D(3,3))
# model.add(Dropout(0.25))
# model.add(Conv2D(128, (3, 3),activation="relu",padding="same"))
# model.add(Conv2D(128, (3, 3),activation="relu",padding="same"))
# model.add(MaxPooling2D(2, 2))
# model.add(Dropout(0.25))
# model.add(Conv2D(128, (3, 3),activation="relu",padding="same"))
# model.add(Conv2D(128, (3, 3),activation="relu",padding="same"))
# model.add(MaxPooling2D(3,3))
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(1024,activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(10,activation="softmax"))

model = Sequential()

#1. KATMAN
model.add(Conv2D(64, 3,activation="relu", data_format="channels_last", kernel_initializer="he_normal", input_shape=(image_size,image_size,3)))
model.add(BatchNormalization())

#2. KATMAN

model.add(Conv2D(64, 3,activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2), strides=2))
model.add(Dropout(0.6)) #%60 unutma işlemi(nöron silme-dropout)

### 3. KATMAN
model.add(Conv2D(128, 3,activation="relu"))
model.add(BatchNormalization())


### 4. KATMAN
model.add(Conv2D(64, 3,activation="relu"))
model.add(BatchNormalization())

### 5. KATMAN
model.add(Conv2D(64, 3,activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(3,3), strides=2))
model.add(Dropout(0.6)) #%60 unutma işlemi(nöron silme-dropout)

### TAM BAĞLANTI KATMANI
model.add(Flatten())
model.add(Dense(1024,activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.6))#%60 unutma işlemi(nöron silme-dropout)

### Çıkış katmanı

model.add(Dense(10,activation="softmax")) #Sınıflama işlemi (7 duygu sınıfı var)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #opmizasyon ve başarım hesaplama metriklerinin belirlenmesi
#model özetini görselleştirelim

model.summary()
# %%
model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["acc"])
history50 = model.fit_generator(
    train,
    epochs=50,
    validation_data=val)
#%%
import numpy as np
from sklearn.metrics import confusion_matrix
Y_pred = model.predict_generator(test,250)
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
# %%
