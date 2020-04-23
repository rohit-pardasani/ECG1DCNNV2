#!/usr/bin/env python

import numpy as np
from tensorflow.keras.models import load_model


def run_12ECG_classifier(data,header_data,classes,model):
    POINTS = 30000
    S = 1000.0
    thresh = 0.5
    Xs = data/S
    Xs = Xs.transpose()
    pts = np.shape(Xs)[0]
    Xi = np.zeros((POINTS,12), dtype=np.float32)
    if(pts>=POINTS):
        Xi = Xs[0:POINTS,:]
    else:
        Xi[0:pts,:] = Xs
    Xi = np.reshape(Xi,(1,POINTS,12))
    Z = model.predict(Xi)
    current_score = Z[0]
    TArr = [0.3786562, 0.36432564, 0.11703539, 0.8391135, 0.17399547, 0.03880799, 0.3140175, 0.65065956, 0.032741368]
    zipPT = zip(current_score,TArr)
    w = [pv>tv for (pv,tv) in zipPT]
    current_label = np.multiply(w,1)
    return current_label,current_score


def load_12ECG_model():
    # load the model from disk 
    filename='ECGCNNX_V4.h5'
    loaded_model = load_model(filename)

    return loaded_model
