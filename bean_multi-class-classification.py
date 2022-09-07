#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 14:41:18 2022

@author: ajaykrishnavajjala
"""
#%%
import tensorflow as tf
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
#%%
data = pd.read_csv("/Users/ajaykrishnavajjala/Documents/School/PHD/Recommender Systems/Tensorflow Course/ANN Multi-class Classification/Dry_Bean_Dataset.csv")
#%%
labels = {'SEKER':0, 'BARBUNYA':1, 'BOMBAY':2, 'CALI':3, 'DERMASON':4, 'HOROZ':5, 'SIRA':6}
data["Class"] = data["Class"].map(labels)
data_labels = data["Class"]
data_features = data.drop("Class", axis=1)
x_train,x_test,y_train,y_test = train_test_split(data_features, data_labels, test_size=0.2, random_state=42)
#%%
N,D = x_train.shape
#%%
bean_model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(D,)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
#%%
bean_model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

#%%
result = bean_model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=10)
#%%
plt.plot(result.history["loss"], label='loss')
plt.plot(result.history['val_loss'], label='val_loss')
plt.legend()
#%%
plt.plot(result.history['accuracy'], label='accuracy')
plt.plot(result.history['val_accuracy'], label='val_accuracy')
plt.legend()

#%%
preds = bean_model.predict(x_test)
evaluation = bean_model.evaluate(x_test, y_test)
#%%
print(evaluation)
print(bean_model.layers)