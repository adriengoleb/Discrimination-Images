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

# ## Neural Network - Approche Deep **AVEC** Data AUGMENTATION

# ### Import des librairies

# +
import pandas as pd
import numpy as np
import os
from glob import glob
import itertools
import fnmatch
import random
import matplotlib.pylab as plt
import seaborn as sns
import cv2
from PIL import Image


import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score

import keras
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D
# %matplotlib inline


import math
from math import floor
# -

# ### Data Preparation

#recherche des fichiers de manière automatisé en fournissant le chemin
from glob import glob
imagePatches = glob('Wang/*.jpg', recursive=True)

imagePatches[:10] #10 premières images récupérées

len(imagePatches) #on retrouve bien les 1000 images de notre base

your_path = 'Wang/145.jpg'
filename = os.path.basename(your_path) #définition de la base du path de nos images
filename = floor( int(filename.split('.')[0]) / 100 ) #récupération du numéro de notre image
filename

x = [] #nos images
y = [] #nos labels
#définition de la taille de nos images
WIDTH = 256
HEIGHT = 256
#pourchaque image de notre chemin
for img in imagePatches:
    filename = os.path.basename(img)
    label = floor( int(filename.split('.')[0]) / 100 ) #récupération du numéro de notre image
    y.append(label)
    
    full_size_image = cv2.imread(img)
    x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC)) 
    #redimensionnement de l'image en suivant une inter-polation cubique
#print(x)

x[0].shape #test taille de notre première image

# ### Learning Strategy

# +
X=np.array(x)
#Normalisation de notre data set
X=X/255.0

### Découpage en apprentissage/test des données
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

# +
# Encodage des labels des iumages  en  "hot vectors" 
#(ex : 5 -> [0,0,0,0,1,0,0,0,0,0])

from keras.utils.np_utils import to_categorical
y_trainHot = to_categorical(Y_train, num_classes = 10)
y_testHot = to_categorical(Y_test, num_classes = 10)


# -

# ### Définition des métriques et évaluations 

class MetricsCheckpoint(Callback):
    """Classe qui sauvegarde les metrics pour chaque epoch. Principe utilisé fréquemment dans 
    les réseaux de neurones pour visualiser l'évolution des étapes d'apprentissage"""
    def __init__(self, savepath):
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}
    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)


# +
#Evolution du taux d'apprentisssage précision par epochs

def plotKerasLearningCurve():
    plt.figure(figsize=(10,5))
    metrics = np.load('logs.npy')[()]
    filt = ['acc'] # try to add 'loss' to see the loss learning curve
    for k in filter(lambda x : np.any([kk in x for kk in filt]), metrics.keys()):
        l = np.array(metrics[k])
        plt.plot(l, c= 'r' if 'val' not in k else 'b', label='val' if 'val' in k else 'train')
        x = np.argmin(l) if 'loss' in k else np.argmax(l)
        y = l[x]
        plt.scatter(x,y, lw=0, alpha=0.25, s=100, c='r' if 'val' not in k else 'b')
        plt.text(x, y, '{} = {:.4f}'.format(x,y), size='15', color= 'r' if 'val' not in k else 'b')   
    plt.legend(loc=4)
    plt.axis([0, None, None, None]);
    plt.grid()
    plt.xlabel('Number of epochs')


# -

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Fonction qui affiche les matrices de confusion
    La normalisation peut être appliquée ou non
    """
    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ### Paramétrage de notre CNN

batch_size = 2
num_classes = 10
epochs = 5
img_rows,img_cols=256,256
input_shape = (img_rows, img_cols, 3)
e = 2

# ### Définition de notre CNN

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,strides=e))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

# ### Data Augmentation

# +
#Augmentation du nombre d'images par la normalisation, la rotation, les décalages, 
#les retournements, le changement de luminosité etc ...

datagen = ImageDataGenerator(
        featurewise_center=False,  #fixe la moyenne d'entrée à 0 sur l'ensemble des données
        rotation_range=20,  # random rotation des images allant de 0 à 180 degrés
        height_shift_range=0.2,  #déplacement aléatoire des images verticalement (fraction de la longueur/hauteur totale)
        width_shift_range=0.2,  # déplacement aléatoire des images horizontalement (fraction de la largeur totale)
        horizontal_flip=True,  # retournement aléatoire des images à l'horizontal
        vertical_flip=True)  # retournement aléatoire des images à la verticale
# -

train= X_train.astype(np.uint8)
train_hot = y_trainHot
test = X_test
test_hot = y_testHot
epochs = 2

# ### Apprentissage et Test

history = model.fit_generator(datagen.flow(train,train_hot, batch_size=5),
                        steps_per_epoch=len(train) / 32, 
                              epochs=epochs,validation_data = [test, test_hot])

plotKerasLearningCurve()
plt.show() 
