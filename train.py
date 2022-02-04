import numpy as np
import cv2
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

train_dir = 'train'
val_dir = 'test'
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

emotion_model = tf.keras.models.Sequential()
emotion_model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size=3, activation='relu', input_shape=(48,48,1)))
emotion_model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size=3, activation='relu'))
emotion_model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
emotion_model.add(tf.keras.layers.Dropout(0.25))
emotion_model.add(tf.keras.layers.Conv2D(filters = 128, kernel_size= 3, activation='relu'))
emotion_model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
emotion_model.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, activation='relu'))
emotion_model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides = 2))
emotion_model.add(tf.keras.layers.Dropout(0.25))
emotion_model.add(tf.keras.layers.Flatten())
emotion_model.add(tf.keras.layers.Dense(1024, activation='relu'))
emotion_model.add(tf.keras.layers.Dropout(0.5))
emotion_model.add(tf.keras.layers.Dense(7, activation='softmax'))
# emotion_model.load_weights('emotion_model.h5')


emotion_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
emotion_model_info = emotion_model.fit(train_generator, epochs=20, validation_data=validation_generator)
emotion_model.save_weights('emotion_model.h5')