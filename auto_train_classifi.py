import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential,Model



def plot_performance(history,initial_epochs=0):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(initial_epochs,len(acc)+initial_epochs)

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show() 


from tensorflow.keras.applications.mobilenet import MobileNet,preprocess_input
img_width,img_height = 224,224
base_model = MobileNet(input_shape=(img_height, img_width, 3),
                         include_top=False,
                         weights='imagenet')

base_model.trainable = False
#base_model.summary()

image_size=(img_height, img_width)
print(image_size)
datagen = ImageDataGenerator(
       validation_split=0.2,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)

train_generator = datagen.flow_from_directory(
        'C:/Users/PTF/Desktop/HM2/programs/dataset',  #path dataset
        subset='training',
        target_size=image_size,
        batch_size=32,
        seed=10,
        class_mode='categorical')
validation_generator = datagen.flow_from_directory(
        'C:/Users/PTF/Desktop/HM2/programs/dataset', #path dataset
        subset='validation',
        target_size=image_size,
        batch_size=32,
        seed=10,
        class_mode='categorical',
        shuffle=False)

print(train_generator.class_indices)
class_names=list(train_generator.class_indices.keys())
class_names

num_classes=train_generator.num_classes

inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = preprocess_input(inputs)
x = base_model(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu',kernel_regularizer='l2')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes,activation='sigmoid',kernel_regularizer='l2')(x)
model = tf.keras.Model(inputs, outputs)


model.summary()

adam=tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=adam,
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

epochs=300
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)

history = model.fit(
  train_generator,
  validation_data=validation_generator,
  epochs=epochs,
  callbacks=[callback]
)

# plot_performance(history)     plot garp

# model.save('model1.h5')
# model.save_weights('weight1.h5')

#FINE TUNE
base_model.trainable = True
fine_tune_at = 10

for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

model.summary()


adam=tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=adam,
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

initial_epochs=5
fine_tune_epochs = 300
total_epochs =  initial_epochs + fine_tune_epochs

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)

history_fine = model.fit(train_generator,
                         validation_data=validation_generator,
                         initial_epoch=initial_epochs,
                         epochs=total_epochs,
                         callbacks=[callback])

# model.save('model2.h5')
# model.save_weights('weight2.h5')

model.save('C:/Users/PTF/Desktop/HM2/model_casification_hm2/model2.h5')
model.save_weights('C:/Users/PTF/Desktop/HM2/model_casification_hm2/weight2.h5')