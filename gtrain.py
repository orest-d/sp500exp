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
parser.add_argument('-i','--input',            help='Inputfile .h5 (rc.h5)',default="rc.h5")
parser.add_argument('-m','--model',            help='Model architecture (gmodel.yaml)',default="gmodel.yaml")
parser.add_argument('-w','--weights',          help='Model weights (gweights.h5)',default="gweights.h5")
parser.add_argument('-b','--batch',            help='Batch size (64)',default="64")
parser.add_argument('-e','--epochs',           help='Number of epochs (1)',default="1")
parser.add_argument('-v','--validation',       help='Validation split (0)',default="0")
parser.add_argument('-r','--regularization',   help='Kernel L2 regularization (0)',default="0")
parser.add_argument('-p','--predict',          help='Prediction file (gpredict.h5)',default="gpredict.h5")

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



class ModelBasis:
    def __init__(self,store, hidden_layers=3, regularization=0.0, batch_size=32, modelfile="gmodel.yaml", weightsfile="gweights.h5",restart=False):
        self.batch_size=batch_size
        self.store=store
        self.regularization = regularization
        self.hidden_layers = hidden_layers
        self.modelfile = modelfile
        self.weightsfile = weightsfile
        self.restart = restart
        self.create_model()        
        self.load_weights()
    def name(self):
        return self.__class__.__name__
    def number_of_outputs(self):
        return 1
    def fit(self,epochs=1):
        training = self.generate(test=False)
        test     = self.generate(test=True)

        self.model.fit_generator(training,
            steps_per_epoch=self.steps_per_epoch,
            epochs=epochs,
            verbose=1,
            validation_data=test,
            validation_steps=self.validation_steps(),
            shuffle=False)
    def save_weights(self):
        self.model.save_weights(self.weightsfile)
    def load_weights(self):
        if self.restart:
            logging.info("Restart; starting with new weigths %s"%self.weightsfile)
        else:
            if exists(self.weightsfile):
                logging.info("Load weights from %s"%self.weightsfile)
                self.model.load_weights(self.weightsfile)
            else:
                logging.info("New weigths %s"%self.weightsfile)
            
class SimpleModelArchitecture:
    def create_model(self):
        logging.info("Create model "+self.name())
        inputs  = self.number_of_inputs()
        outputs = self.number_of_outputs()
        model = Sequential()
        L=int(1.5*max(inputs,outputs))
        logging.info("Inputs:                  %d"%inputs)
        logging.info("Outputs:                 %d"%outputs)
        logging.info("Hidden layer size:       %d"%L)
        logging.info("Number of hidden layers: %d"%self.hidden_layers)
        
        regularization=self.regularization

        model.add(Dense(units=L, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization), input_dim=inputs))
        for i in range(self.hidden_layers):
            model.add(Dense(units=L, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization), input_dim=L))
        model.add(Dense(units=outputs, activation='linear', use_bias=True, kernel_regularizer=regularizers.l2(regularization), input_dim=L))
        #model.add(Activation('relu'))
        model.compile(optimizer='rmsprop',loss='mse')
        modelfile = self.modelfile
        logging.info("Save model to %s"%modelfile)
        with open(modelfile,"w") as f:
            f.write(model.to_yaml())
        self.model = model
        return self

class Sigma1(ModelBasis,SimpleModelArchitecture):
    test_days=10
    steps_per_epoch=100000
    def number_of_inputs(self):
        days=len(store["Days_scaled"].columns)
        names=len(store["Names_scaled"].columns)        
        return days+names
    def validation_steps(self):
        return int(len(store["Names_scaled"].columns)*self.test_days/self.batch_size)        
    def generate(self,test=False):
        store=self.store
        batch_size=self.batch_size
        returns=store["Returns_scaled"]
        days=store["Days_scaled"]
        names=store["Names_scaled"]
        if test:
            day_order=np.arange(len(days)-self.test_days)
        else:
            day_order=len(days)-np.arange(self.test_days)-1
        names_order=np.arange(len(names))
        nb = int(len(names)/batch_size)
        r=returns.as_matrix()
        ddm=days.as_matrix()
        ndm=names.as_matrix()
        while True:
            np.random.shuffle(day_order)
            for i in day_order:
                dv = ddm[i]
                rv=r[i]
                np.random.shuffle(names_order)
                for j in range(nb):                
                    name_batch_index=names_order[j*batch_size:(j+1)*batch_size]
                    r_batch = rv[name_batch_index]
                    names_batch=ndm[name_batch_index]
                    X=np.concatenate((np.broadcast_to(dv,(batch_size,len(dv))),names_batch),axis=1)
                    Y=(r_batch*r_batch).reshape((batch_size,1))
                    yield X,Y


m=Sigma1(store)
m.fit()