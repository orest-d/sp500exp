import pandas as pd
import numpy as np
import logging
from os.path import exists
from sklearn import preprocessing
import argparse
import sys
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import model_from_yaml


parser = argparse.ArgumentParser(description='Preprocess data.')
parser.add_argument('-i','--input',    help='Inputfile .csv (train.h5)',default="train.h5")
parser.add_argument('-m','--model',    help='Model architecture (model.yaml)',default="model.yaml")
parser.add_argument('-w','--weights',  help='Model weights (weights.h5)',default="weights.h5")
parser.add_argument('-b','--batch',    help='Batch size (32)',default="32")
parser.add_argument('-e','--epochs',   help='Number of epochs (50)',default="50")



args = parser.parse_args()
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, filename='log.txt')

inputfile=args.input
modelfile=args.model
weightsfile=args.weights
batch = int(args.batch)
epochs = int(args.epochs)

logging.info("Input:        "+inputfile)
logging.info("Model:        "+modelfile)
logging.info("Weights:      "+weightsfile)
logging.info("Batch:        "+str(batch))
logging.info("Epochs:       "+str(epochs))

store = pd.HDFStore(inputfile)
X_training=store["X_training"]
Y_training=store["Y_training"]
inputs = len(X_training.columns)
outputs = len(Y_training.columns)

if exists(modelfile):
    logging.info("Load model from %s"%modelfile)
    model = model_from_yaml(open(modelfile).read())
    model.compile(optimizer='rmsprop',loss='mse')
else:
    logging.info("Create default model")
    model = Sequential()
    L=int(1.8*inputs)
    model.add(Dense(units=L, activation='relu', use_bias=True, input_dim=inputs))
    model.add(Dense(units=L, activation='relu', use_bias=True, input_dim=L))
    model.add(Dense(units=L, activation='relu', use_bias=True, input_dim=L))
    model.add(Dense(units=outputs, activation='linear', use_bias=True, input_dim=L))
    #model.add(Activation('relu'))
    model.compile(optimizer='rmsprop',loss='mse')
    logging.info("Save model to %s"%modelfile)
    with open(modelfile,"w") as f:
        f.write(model.to_yaml())

if exists(weightsfile):
    logging.info("Load weights from %s"%weightsfile)
    model.load_weights(weightsfile)

X=X_training.as_matrix()
Y=Y_training.as_matrix()

print ()
if epochs>0:
    history=model.fit(X,Y, batch_size=batch, epochs=epochs, verbose=1)
    print(history.history)
    model.save_weights(weightsfile)

print ("Evaluate")

eval_training=model.evaluate(X,Y)
X=store["X_test"].as_matrix()
Y=store["Y_test"].as_matrix()
eval_test=model.evaluate(X,Y)
print ("Training: "+str(eval_training))
print ("Test:     "+str(eval_test))
logging.info("Evaluate training:%f test:%f"%(eval_training,eval_test))
