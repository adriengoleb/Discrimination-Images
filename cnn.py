# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Neural Network - Approche Deep **SANS** Data AUGMENTATION

# ### Import des librairies

from __future__ import print_function
# Télécharger Keras 
import keras
# Télécharger le modèle séquentiel 
from keras.models import Sequential
# Télécharger les couches des cellules neuronales 
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix
import numpy as np
import os


# ### Lecture de nos images

def read_image(path, img_rows, img_cols):
    img = image.load_img(path, target_size=(img_rows, img_cols))
    tmp = image.img_to_array(img)
    tmp = np.expand_dims(tmp, axis=0)
    #tmp = preprocess_input(tmp)
    return tmp


#charger liste des images
img_name = list()
for i in range(0,1000):
    img_name.append(str(i)+".jpg")

# ### Paramétrage

# +
#vérité fondamentale
target = np.array([0]*100+[1]*100+[2]*100+[3]*100+[4]*100+[5]*100+[6]*100+[7]*100+[8]*100+[9]*100)

# Nombre de caractéristiques de données différentes : chiffres 0-9
num_classes = 10
# Nombre de périodes pour la formation du réseau de neurones
epochs = 1
# Nombre de blocs de données utilisée lors d'un passage/epoch
batch_size = 5
# Dimensions des images d'entrée (28 x 28 pixels par image)
img_rows, img_cols = 256, 256

# +
# Télécharger les formations et les tests

#img_path = 'Wang/*.jpg'
img_path = r'C:\Users\adrien\Desktop\EICNAM\3ème année\ML\Projet 1 - Réseaux de neurones\Wang\0.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)

#initialisation des train et test_images
train_images = img_batch
test_images = img_batch
train_labels = target
test_labels = target

# +
#Constitution de nos batch de données images et mise à jour de nos données "train" et "test"

batch_holder = np.zeros((1000, img_rows, img_cols, 3))
i = 0
for name in img_name:
    batch_holder[i, :] = read_image(name, img_rows, img_cols)[0]
    i = i+1

train_images = batch_holder
test_images = batch_holder

input_shape = (img_rows, img_cols, 3)
# -

# ### Transformation des features

# +

# Définition du type de données en nombre décimal 
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

# Normalisation des données des images
train_images /= 255
test_images /= 255
print('train_images shape:', train_images.shape)
print(train_images.shape[0], 'train samples')
print(test_images.shape[0], 'test samples')

# Convertion des vecteurs de classe en matrices de classe binaires
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)
# -

# ### Définition de notre modèle

# +
# Création du modèle

model = Sequential()

# Ajouter des couches au modèle
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compilation du modèle
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# -

# ### Apprentissage et Test

#Apprentisssage du modèle
model.fit(train_images, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(test_images, test_labels))

# Évaluation du modèle
predictions = model.predict(test_images, batch_size = batch_size, verbose=0)
#affichage des résultats (matrice de confusion et score de précision/accuracy)
pred = list()
for i in range(len(predictions)) :
    pred.append(np.argmax(predictions[i]))
confusion_matrix(target, pred)
