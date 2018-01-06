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
from keras import regularizers

parser = argparse.ArgumentParser(description='Training.')
parser.add_argument('-i','--input',            help='Inputfile .h5 (train.h5)',default="train.h5")
parser.add_argument('-m','--model',            help='Model architecture (model.yaml)',default="model.yaml")
parser.add_argument('-w','--weights',          help='Model weights (weights.h5)',default="weights.h5")
parser.add_argument('-b','--batch',            help='Batch size (32)',default="32")
parser.add_argument('-e','--epochs',           help='Number of epochs (50)',default="50")
parser.add_argument('-v','--validation',       help='Validation split (0)',default="0")
parser.add_argument('-r','--regularization',   help='Kernel L2 regularization (0)',default="0")
parser.add_argument('-p','--predict',          help='Prediction file (predict.h5)',default="predict.h5")

args = parser.parse_args()
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, filename='log.txt')

inputfile=args.input
modelfile=args.model
weightsfile=args.weights
batch = int(args.batch)
epochs = int(args.epochs)
validation_split = float(args.validation)
regularization = float(args.regularization)
predict = args.predict

logging.info("Input:                     "+inputfile)
logging.info("Model:                     "+modelfile)
logging.info("Weights:                   "+weightsfile)
logging.info("Batch:                     "+str(batch))
logging.info("Epochs:                    "+str(epochs))
logging.info("Validation split:          "+str(validation_split))
logging.info("Kernel L2 regularization:  "+str(regularization))
logging.info("Prediction file:           "+predict)

store = pd.HDFStore(inputfile)
X_training=store["X_training"]
Y_training=store["Y_training"]
inputs = len(X_training.columns)
outputs = len(Y_training.columns)

if False:#exists(modelfile):
    logging.info("Load model from %s"%modelfile)
    model = model_from_yaml(open(modelfile).read())
    model.compile(optimizer='rmsprop',loss='mse')
else:
    logging.info("Create default model")
    model = Sequential()
    L=int(1.5*max(inputs,outputs))
    model.add(Dense(units=L, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization), input_dim=inputs))
    model.add(Dense(units=L, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization), input_dim=L))
    model.add(Dense(units=L, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization), input_dim=L))
#    model.add(Dense(units=L, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization), input_dim=L))
    model.add(Dense(units=L, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization), input_dim=L))
    model.add(Dense(units=outputs, activation='linear', use_bias=True, kernel_regularizer=regularizers.l2(regularization), input_dim=L))
    #model.add(Activation('relu'))
    model.compile(optimizer='rmsprop',loss='mse')
    logging.info("Save model to %s"%modelfile)
    with open(modelfile,"w") as f:
        f.write(model.to_yaml())

if exists(weightsfile):
    logging.info("Load weights from %s"%weightsfile)
    model.load_weights(weightsfile)

X1=X_training.as_matrix()
Y1=Y_training.as_matrix()
X2=store["X_test"].as_matrix()
Y2=store["Y_test"].as_matrix()

if epochs>0:
    if validation_split>0:
        validation_data=None
    else:
        validation_data=(X2,Y2)
    history=model.fit(X1,Y1, validation_data=validation_data, validation_split=validation_split, batch_size=batch, epochs=epochs, verbose=1)
    print(history.history)
    model.save_weights(weightsfile)

print ("Evaluate")

eval_training=model.evaluate(X1,Y1)
eval_test=model.evaluate(X2,Y2)
print ("Training: "+str(eval_training))
print ("Test:     "+str(eval_test))
logging.info("Evaluate training:%f test:%f"%(eval_training,eval_test))

def scale_df(df):
    return pd.DataFrame(data=[dict(df.mean()),dict(df.std())],columns=df.columns,index=["mean","std"])

def trivial_scale_df(df):
    return pd.DataFrame(data={c:(0,1) for c in df.columns},columns=df.columns,index=["mean","std"])

def rescale_df(df,scale):
    return (df-scale.loc["mean"])/scale.loc["std"]

def inverse_scale_df(df,scale):
    return df*scale.loc["std"]+scale.loc["mean"]

if len(predict)>2:
    logging.info("Prediction")
    predict_store = pd.HDFStore(predict)

    for key in store.keys():
        predict_store[key]=store[key]
    
    x_scale = store["X_scale"]
    y_scale = store["Y_scale"]
    X=rescale_df(store["X"],x_scale)
    Y=store["Y"]
    P=inverse_scale_df(pd.DataFrame(data=model.predict(X.as_matrix(), batch_size=batch),columns=Y.columns,index=Y.index),y_scale)
    predict_store["P"]=P

    X=store["X_training"]
    Y=store["Y_training"]
    P=pd.DataFrame(data=model.predict(X.as_matrix(), batch_size=batch),columns=Y.columns,index=Y.index)
    predict_store["P_training"]=P

    X=store["X_test"]
    Y=store["Y_test"]
    P=pd.DataFrame(data=model.predict(X.as_matrix(), batch_size=batch),columns=Y.columns,index=Y.index)
    predict_store["P_test"]=P

else:
    logging.info("Prediction skipped")
