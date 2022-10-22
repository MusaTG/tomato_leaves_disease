# %%

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
from tensorflow import keras
# image settings

image_size=64

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

# reading image

training_data = train_datagen.flow_from_directory(
    "Tomato_Dataset\\train",
    target_size=(image_size,image_size),
    batch_size=64,
    class_mode='categorical' 
)

# print(training_data.class_indices)

testing_data = test_datagen.flow_from_directory(
    "Tomato_Dataset\\val",
    target_size=(image_size,image_size),
    batch_size=64,
    class_mode='categorical'
)

# print(testing_data.image_shape)
# print(testing_data.)

# %%

# model

model = Sequential()
model.add(Conv2D(32,(3,3),activation="relu", input_shape =(image_size,image_size,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128,(3,3),activation="relu"))

model.add(Flatten())

model.add(Dense(128,activation = "relu"))
model.add(Dense(10,activation = "softmax"))

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["acc"])

model.summary()


history = model.fit_generator(
        training_data,
        steps_per_epoch=100,
        epochs=12,
        validation_data=testing_data,
        validation_steps=30)



# %%
import matplotlib as plt

acc_original=history.history["acc"]
val_acc_original=history.history["val_acc"]
loss_original=history.history["loss"]
val_loss_original=history.history["val_loss"]

epochsO=range(1,len(acc_original)+1)
val_epochsO=range(1,len(val_acc_original)+1)


sizeplt=plt.figure()
sizeplt.set_figwidth(16)
sizeplt.set_figheight(6)

plt.plot(epochsO,acc_original,'mo',label="Eğitim başarımı(original)")
plt.plot(val_epochsO,val_acc_original,'m--',label="Doğrulama başarımı(original)")
plt.title("Eğitim ve Doğrulama başarımı")
plt.legend()
plt.figure()


sizeplt=plt.figure()
sizeplt.set_figwidth(16)
sizeplt.set_figheight(6)

plt.plot(epochsO,loss_original,'ko',label="Eğitim kaybı(original)")
plt.plot(val_epochsO,val_loss_original,'m--',label="Doğrulama kaybı(original)")
plt.title("Eğitim ve Doğrulama kaybı")
plt.legend()

plt.show()
