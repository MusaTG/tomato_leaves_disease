# %%
from keras_preprocessing.image import ImageDataGenerator

image_size = 128
batch_size = 8

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
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
    model = Sequential()
    model.add(Conv2D(128,(3,3),activation="relu",input_shape=(image_size,image_size,3)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3),activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(3,3))
    model.add(Dropout(0.6))

    model.add(Conv2D(256, (3, 3),activation="relu"))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3),activation="relu"))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3),activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(3,3))
    model.add(Dropout(0.6))

    model.add(Flatten())
    model.add(Dense(1024,activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))

    model.add(Dense(10,activation="softmax"))

    model.summary()

    return model

# %%
model = Model()
model.compile(optimizer="RMSProp",loss="categorical_crossentropy",metrics=["acc"])
history50 = model.fit_generator(
    train,
    epochs=100,
    validation_data=val)

# %%
model.save("model_100.h5")

# %%
import visualkeras as vk
from PIL import ImageFont
font = ImageFont.truetype("arial.ttf", 18) 
vk.layered_view(model,legend=True,draw_volume=False, to_file="modeldrawFalse.png",font=font).show()


# %%
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report

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
clr = classification_report(ytrue, ypred, target_names=list(test.class_indices.keys()), digits= 4) # create classification report
print("Classification Report:\n----------------------\n", clr)

# %%
import matplotlib.pyplot as plt
acc_original=history50.history["acc"]
val_acc_original=history50.history["val_acc"]
loss_original=history50.history["loss"]
val_loss_original=history50.history["val_loss"]

epochs=range(1,len(acc_original)+1)
val_epochs=range(1,len(val_acc_original)+1)


sizeplt=plt.figure()
sizeplt.set_figwidth(16)
sizeplt.set_figheight(6)

plt.plot(epochs,acc_original,'bo',label="Eğitim başarımı")
plt.plot(val_epochs,val_acc_original,'b--',label="Doğrulama başarımı")
plt.title("Eğitim ve Doğrulama Başarımı")
plt.legend()
plt.figure()


sizeplt=plt.figure()
sizeplt.set_figwidth(16)
sizeplt.set_figheight(6)

plt.plot(epochs,loss_original,'ro',label="Eğitim kaybı")
plt.plot(val_epochs,val_loss_original,'r--',label="Doğrulama kaybı")
plt.title("Eğitim ve Doğrulama Kaybı")
plt.legend()

plt.show()
# %%
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite","wb") as f:
    f.write(tflite_model)

# %% control

interpreter = tf.lite.Interpreter(model_path="model.tflite")
input_details=interpreter.get_input_details()
output_details=interpreter.get_output_details()

print("Input shape:",input_details[0]["shape"])
print("Input type:",input_details[0]["dtype"])
print("Output shape:",output_details[0]["shape"])
print("Output type:",output_details[0]["dtype"])

# %%
from keras.models import load_model
model = load_model('model_100.h5')
# %%
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# %%
