# %%

from keras.preprocessing.image import ImageDataGenerator
# image settings

image_size=32
batch_size=32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=.2,
    height_shift_range=.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
    )

test_datagen = ImageDataGenerator(rescale=1./255)

# reading image

training_data = train_datagen.flow_from_directory(
    "Tomato_Dataset\\train_orj",
    target_size=(image_size,image_size),
    batch_size=batch_size,
    class_mode='categorical' 
)

# print(training_data.class_indices)

testing_data = test_datagen.flow_from_directory(
    "Tomato_Dataset\\val",
    target_size=(image_size,image_size),
    batch_size=batch_size,
    class_mode='categorical'
)

# print(testing_data.image_shape)
# print(testing_data.)

# %%

# model

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

model = Sequential()
model.add(Conv2D(64,(3,3),activation="relu", input_shape =(image_size,image_size,3)))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128,(3,3),activation="relu"))
model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPooling2D(2,2))


model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512,activation = "relu"))
model.add(Dense(10,activation = "softmax"))

model.compile(optimizer="RMSprop",loss="categorical_crossentropy",metrics=["acc"])

model.summary()

# %%
history = model.fit_generator(
        training_data,
        steps_per_epoch=312,
        epochs=12,
        validation_data=testing_data,
        validation_steps=30)



# %%
import matplotlib.pyplot as plt
acc_original=history.history["acc"]
val_acc_original=history.history["val_acc"]
loss_original=history.history["loss"]
val_loss_original=history.history["val_loss"]

epochs=range(1,len(acc_original)+1)
val_epochs=range(1,len(val_acc_original)+1)


sizeplt=plt.figure()
sizeplt.set_figwidth(16)
sizeplt.set_figheight(6)

plt.plot(epochs,acc_original,'mo',label="Eğitim başarımı(original)")
plt.plot(val_epochs,val_acc_original,'m--',label="Doğrulama başarımı(original)")
plt.title("Eğitim ve Doğrulama başarımı")
plt.legend()
plt.figure()


sizeplt=plt.figure()
sizeplt.set_figwidth(16)
sizeplt.set_figheight(6)

plt.plot(epochs,loss_original,'ko',label="Eğitim kaybı(original)")
plt.plot(val_epochs,val_loss_original,'m--',label="Doğrulama kaybı(original)")
plt.title("Eğitim ve Doğrulama kaybı")
plt.legend()

plt.show()

# %%

loss, accuracy=model.evaluate_generator(testing_data,len(testing_data))
print(str(loss)+"\n"+str(accuracy))
print(len(testing_data))

#%%
import tensorflow as tf
tf.__version__
# %%

# model

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

model = Sequential()
model.add(Conv2D(64,(3,3),activation="relu", input_shape =(image_size,image_size,3)))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128,(3,3),activation="relu"))
model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPooling2D(2,2))


model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512,activation = "relu"))
model.add(Dense(10,activation = "softmax"))

model.compile(optimizer="Adadelta",loss="categorical_crossentropy",metrics=["acc"])

model.summary()

# %%

# model

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

model = Sequential()
model.add(Conv2D(64,(3,3),activation="relu", input_shape =(image_size,image_size,3)))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128,(3,3),activation="relu"))
model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPooling2D(2,2))


model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512,activation = "relu"))
model.add(Dense(10,activation = "softmax"))

model.compile(optimizer="SGD",loss="categorical_crossentropy",metrics=["acc"])

model.summary()

# %%

# model

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

model = Sequential()
model.add(Conv2D(64,(3,3),activation="relu", input_shape =(image_size,image_size,3)))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128,(3,3),activation="relu"))
model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPooling2D(2,2))


model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512,activation = "relu"))
model.add(Dense(10,activation = "softmax"))

model.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["acc"])

model.summary()

#%%
print("RMSprop")
# %%
# model

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

model = Sequential()
model.add(Conv2D(64,(3,3),activation="tanh", input_shape =(image_size,image_size,3)))
model.add(Conv2D(64,(3,3),activation="tanh"))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128,(3,3),activation="tanh"))
model.add(Conv2D(128,(3,3),activation="tanh"))
model.add(MaxPooling2D(2,2))


model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512,activation = "tanh"))
model.add(Dense(10,activation = "softmax"))

model.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["acc"])

model.summary()

#%%
print("Adam and tanh")

# %%
# model

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

model = Sequential()
model.add(Conv2D(64,(3,3),activation="tanh", input_shape =(image_size,image_size,3)))
model.add(Conv2D(64,(3,3),activation="tanh"))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128,(3,3),activation="tanh"))
model.add(Conv2D(128,(3,3),activation="tanh"))
model.add(MaxPooling2D(2,2))


model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512,activation = "tanh"))
model.add(Dense(10,activation = "softmax"))

model.compile(optimizer="RMSprop",loss="categorical_crossentropy",metrics=["acc"])

model.summary()

#%%
print("RMSprop and tanh")
# %%
