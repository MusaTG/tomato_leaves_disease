#%%
from keras_preprocessing.image import ImageDataGenerator

image_size = 64
batch_size = 8

datagen = ImageDataGenerator(
    rescale=1./255
    # ,
    # validation_split=0.2
)

train = datagen.flow_from_directory(
    "Vegetable Images\\train",
    # subset="training",
    target_size=(image_size,image_size),
    batch_size=batch_size,
    class_mode="categorical"
)

val = datagen.flow_from_directory(
    "Vegetable Images\\validation",
    target_size=(image_size,image_size),
    batch_size=batch_size,
    # subset="validation",
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test = test_datagen.flow_from_directory(
    "Vegetable Images\\test",
    target_size=(image_size,image_size),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical'
)
# %%
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization,Input
from keras.models import Sequential
import tensorflow as tf

def Model():
    model = Sequential()
    model.add(Conv2D(256,(3,3),activation="relu",input_shape=(image_size,image_size,3)))
    model.add(Conv2D(256, (3, 3),activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(3,3))
    model.add(Conv2D(128, (3, 3),activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.6))
    model.add(Conv2D(64, (3, 3),activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(3,3))
    model.add(Dropout(0.6))
    model.add(Flatten())
    model.add(Dense(512,activation="relu"))
    model.add(Dropout(0.6))
    model.add(Dense(15,activation="softmax"))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #opmizasyon ve başarım hesaplama metriklerinin belirlenmesi

    model.summary()

    return model

model = Model()
# Sequential([
#     Conv2D(64,(3,3),activation="relu",padding="same",input_shape=[image_size,image_size,3]),
#     BatchNormalization(),
#     MaxPooling2D(3,3),
#     Conv2D(128,(3,3),activation="relu",padding="same"),
#     BatchNormalization(),
#     Conv2D(128,(3,3),activation="relu",padding="same"),
#     BatchNormalization(),
#     MaxPooling2D(3,3),
#     Conv2D(256,(3,3),activation="relu",padding="same"),
#     BatchNormalization(),
#     Conv2D(256,(3,3),activation="relu",padding="same"),
#     BatchNormalization(),
#     MaxPooling2D(3,3),
#     Flatten(),
#     Dense(128,activation="relu"),
#     BatchNormalization(),
#     Dropout(.5),
#     Dense(64,activation="relu"),
#     BatchNormalization(),
#     Dropout(.5),
#     Dense(10,activation="softmax")
# ])


# model.add(Conv2D(128,(3,3),activation="relu",input_shape=(image_size,image_size,3)))
# model.add(BatchNormalization())

# model.add(Conv2D(128, (3, 3),activation="relu"))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(3,3))
# model.add(Dropout(0.6))

# model.add(Conv2D(128, (3, 3),activation="relu"))
# model.add(BatchNormalization())

# model.add(Conv2D(128, (3, 3),activation="relu"))
# model.add(BatchNormalization())

# model.add(Conv2D(128, (3, 3),activation="relu"))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(3,3))
# model.add(Dropout(0.6))

# model.add(Flatten())
# model.add(Dense(1024,activation="relu"))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Dense(10,activation="softmax"))

# model = Sequential()

# #1. KATMAN
# model.add(Conv2D(64, 3,activation="relu", data_format="channels_last", kernel_initializer="he_normal", input_shape=(image_size,image_size,3)))
# model.add(BatchNormalization())

# #2. KATMAN

# model.add(Conv2D(64, 3,activation="relu"))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2), strides=2))
# model.add(Dropout(0.6)) #%60 unutma işlemi(nöron silme-dropout)

# ### 3. KATMAN
# model.add(Conv2D(128, 3,activation="relu"))
# model.add(BatchNormalization())


# ### 4. KATMAN
# model.add(Conv2D(64, 3,activation="relu"))
# model.add(BatchNormalization())

# ### 5. KATMAN
# model.add(Conv2D(64, 3,activation="relu"))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(3,3), strides=2))
# model.add(Dropout(0.6)) #%60 unutma işlemi(nöron silme-dropout)

# ### TAM BAĞLANTI KATMANI
# model.add(Flatten())
# model.add(Dense(1024,activation="relu"))
# model.add(BatchNormalization())
# model.add(Dropout(0.6))#%60 unutma işlemi(nöron silme-dropout)

# ### Çıkış katmanı

# model.add(Dense(10,activation="softmax")) #Sınıflama işlemi (7 duygu sınıfı var)

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #opmizasyon ve başarım hesaplama metriklerinin belirlenmesi
#model özetini görselleştirelim

# model.summary()

# %%
# model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["acc"])
history50 = model.fit_generator(
    train,
    epochs=50,
    verbose=1,
    validation_data=val)
# %%
model.evaluate_generator(test)
#%%
import numpy as np
from sklearn.metrics import confusion_matrix
Y_pred = model.predict_generator(test)
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
clr = classification_report(ytrue, ypred, digits= 4) # create classification report
print("Classification Report:\n----------------------\n", clr)
# %%
model.save("deneme2EminBorandag.h5")
# %%
yy = []
yt = test.labels
classes = list(test.class_indices.keys())