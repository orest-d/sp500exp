import pandas as pd
import numpy as np
import logging
from os.path import exists
from sklearn import preprocessing
import argparse
import sys
from keras.models import Sequential,Model
from keras.layers import Dense, Activation, Input, concatenate, BatchNormalization, multiply,dot
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
parser.add_argument('-P','--predict',          help='Prediction file (gpredict.h5)',default="gpredict.h5")
parser.add_argument('-p','--process',          help='Process (Sigma1)',default="Sigma1")

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
process = args.process

logging.info("Input:                     "+inputfile)
logging.info("Model:                     "+modelfile)
logging.info("Weights:                   "+weightsfile)
logging.info("Batch:                     "+str(batch))
logging.info("Epochs:                    "+str(epochs))
logging.info("Validation split:          "+str(validation_split))
logging.info("Kernel L2 regularization:  "+str(regularization))
logging.info("Prediction file:           "+predict)
logging.info("Process:                   "+process)

store = pd.HDFStore(inputfile)



class ModelBasis:
    def __init__(self,store, regularization=0.0, batch_size=32, modelfile="gmodel.yaml", weightsfile="gweights.h5",restart=False):
        self.batch_size=batch_size
        self.store=store
        self.regularization = regularization
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
                self.model.load_weights(self.weightsfile, by_name=True)
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
    hidden_layers=3
    def number_of_inputs(self):
        days=len(store["Days_scaled"].columns)
        names=len(store["Names_scaled"].columns)        
        return days+names
    def validation_steps(self):
        return int(len(store["Names_scaled"].columns)*self.test_days/self.batch_size)        
    def generate(self,test=False):
        store=self.store
        batch_size=self.batch_size
        returns=store["Returns"]
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
        cov=store["Covariance"].as_matrix()
        dcov=cov.diagonal()
        dcovstd=dcov.std()

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
                    Y=((r_batch*r_batch-dcov[name_batch_index])/dcovstd).reshape((batch_size,1))
                    yield X,Y

class Cov1(ModelBasis,SimpleModelArchitecture):
    test_days=10
    steps_per_epoch=100000
    hidden_layers=2
    def number_of_inputs(self):
        days=len(store["Days_scaled"].columns)
        names=len(store["Names_scaled"].columns)        
        return days+names+names
    def validation_steps(self):
        return int(len(store["Names_scaled"].columns)*self.test_days/self.batch_size)        
    def generate(self,test=False):
        store=self.store
        batch_size=self.batch_size
        returns=store["Returns"]
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
        cov=store["Covariance"].as_matrix()
        dcov=cov.diagonal()
        dcovstd=dcov.std()

        while True:
            np.random.shuffle(day_order)
            for i in day_order:
                dv = ddm[i]
                rv=r[i]
#                for k in range(len(names)):
                j=0
                np.random.shuffle(names_order)
                name_batch_index=names_order[j*batch_size:(j+1)*batch_size]
                k=np.random.choice(names_order)
                while k in name_batch_index:
                    k=np.random.choice(names_order)

                rk=rv[k]
                name_k=ndm[k]
#                for j in range(nb):

                name_batch_index=names_order[j*batch_size:(j+1)*batch_size]
                r_batch = rv[name_batch_index]
                names_batch=ndm[name_batch_index]
                dv_batch=np.broadcast_to(dv,(batch_size,len(dv)))
                name_k_batch=np.broadcast_to(name_k,(batch_size,len(name_k)))
                X=np.concatenate((dv_batch,name_k_batch,names_batch),axis=1)
                c=cov[k,name_batch_index]
                Y=((rk*r_batch-c)/c).reshape((batch_size,1))
                print (Y.mean(),Y.std())
                yield X,Y

class DayNameNameIndexGenerator:
    def index_generator(self,test=False):
        days=store["Days_scaled"]
        names=store["Names_scaled"]
        if test:
            day_order=len(days)-np.arange(self.test_days)-1-self.horizon
        else:
            day_order=np.arange(self.horizon,len(days)-self.test_days-2*self.horizon)

        names_order1=np.arange(len(names))
        names_order2=np.arange(len(names))
        while True:            
            np.random.shuffle(day_order)
            np.random.shuffle(names_order1)
            np.random.shuffle(names_order2)
            for a,b,c in zip(day_order,names_order1,names_order2):
                if b==c:
                    continue
                yield a,b,c
                yield a,c,b
    def index_batch_generator(self,test=False):
        g=self.index_generator(test)
        while True:
            a_batch=[]
            b_batch=[]
            c_batch=[]
            for i in range(self.batch_size):
                a,b,c=next(g)
                a_batch.append(a)
                b_batch.append(b)
                c_batch.append(c)
            yield np.array(a_batch),np.array(b_batch),np.array(c_batch)

class Cov2(ModelBasis,SimpleModelArchitecture,DayNameNameIndexGenerator):
    test_days=10
    steps_per_epoch=100000
    hidden_layers=2
    horizon=10
    def number_of_inputs(self):
        days=len(store["Days_scaled"].columns)
        names=len(store["Names_scaled"].columns)        
        return days+names+names
    def validation_steps(self):
        return int(len(store["Names_scaled"].columns)*self.test_days/self.batch_size)

    def generate(self,test=False):
        store=self.store
        batch_size=self.batch_size
        returns=store["Returns_scaled"]
        days=store["Days_scaled"]
        names=store["Names_scaled"]


        r=returns.as_matrix()
        ddm=days.as_matrix()
        ndm=names.as_matrix()
        cov=store["Covariance"].as_matrix()
        dcov=cov.diagonal()
        dcovstd=dcov.std()
        
        ibg=self.index_batch_generator(test)
        while True:
            day_index, n1_index, n2_index = next(ibg)
            d=ddm[day_index,:]
            n1=ndm[n1_index,:]
            n2=ndm[n2_index,:]
            X=np.concatenate((d,n1,n2),axis=1)
            Y=np.zeros(self.batch_size,np.float32)
            for i in range(self.batch_size):
                c=cov[n1_index[i],n2_index[i]]
                begin=0#int(-self.horizon/4)
                for j in range(begin,begin+self.horizon):
                    Y[i]+=r[day_index[i]+j,n1_index[i]]*r[day_index[i]+j,n2_index[i]]
                Y[i]=Y[i]/self.horizon

            Y=Y.reshape((batch_size,1))
            #print (Y.mean(),Y.std())
            yield X,Y


class F1ModelArchitecture:
    stock_history_length=100
    def create_model(self):
        logging.info("Create model "+self.name())
        store=self.store
        returns=store["Returns_scaled"]
        market_size=len(returns.columns)

        stock_L1=int(1.5*self.stock_history_length)
        stock_Lf=20
        market_L1=int(1.5*market_size)
        market_Lf=stock_Lf
        regularization=self.regularization

        stock_inputs=Input(shape=(self.stock_history_length,),name="stock_inputs")
        x=Dense(name="stock_layer1",units=stock_L1, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(stock_inputs)
        x=BatchNormalization()(x)
        x=Dense(name="stock_layer2",units=stock_L1, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(x)
        x=BatchNormalization()(x)
        x=Dense(name="stock_layer3",units=stock_L1, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(x)
        stock_fingerprint=Dense(name="stock_fingerprint",units=stock_Lf, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(x)

        market_inputs=Input(shape=(market_size,),name="market_inputs")
        x=Dense(name="market_layer1",units=market_L1, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(market_inputs)
        #x=BatchNormalization()(x)
        x=Dense(name="market_layer2",units=market_L1, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(x)
        #x=BatchNormalization()(x)
        x=Dense(name="market_layer3",units=market_L1, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(x)
        market_fingerprint=Dense(name="market_fingerprint",units=market_Lf, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(x)

        mul=multiply([stock_fingerprint,market_fingerprint])
        fingerprints = concatenate([stock_fingerprint,market_fingerprint,mul])
        x=Dense(name="combine_layer1",units=100, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(fingerprints)
        #x=BatchNormalization()(x)
        x=Dense(name="combine_layer2",units=60, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(x)
        #x=BatchNormalization()(x)
        x=Dense(name="combine_layer3",units=60, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(x)
        output=Dense(units=1, activation='linear', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(x)
        model = Model(inputs=[stock_inputs,market_inputs],outputs=output)
        model.compile(optimizer='rmsprop',loss='mse')
        modelfile = self.modelfile
        logging.info("Save model to %s"%modelfile)
        with open(modelfile,"w") as f:
            f.write(model.to_yaml())
        self.model = model
        return self

class DayDayNameF1IndexGenerator:
    def index_generator(self,test=False):
        H=self.stock_history_length+2
        r=store["Returns_scaled"].as_matrix()
        N,M=r.shape

        if test:
            day_order=N-np.arange(self.test_days)-H-2
        else:
            day_order=np.arange(N-self.test_days-2*H-1)

        stocks=np.arange(M)
        
        while True:
            np.random.shuffle(day_order)
            for period_start_day in day_order:
                prediction_days=period_start_day+H-1+np.arange(3)
                np.random.shuffle(stocks)
                
                pd_repeated=np.tile(prediction_days,M)
                stocks_repeated=np.repeat(stocks,len(prediction_days))
                o=np.concatenate((pd_repeated,stocks_repeated)).reshape((2,len(prediction_days)*len(stocks))).T
                np.random.shuffle(o)
                for prediction_day,stock_number in o[:2]:
                    yield period_start_day,prediction_day,stock_number

    def index_batch_generator(self,test=False):
        g=self.index_generator(test)
        while True:
            a_batch=[]
            b_batch=[]
            c_batch=[]
            for i in range(self.batch_size):
                a,b,c=next(g)
                a_batch.append(a)
                b_batch.append(b)
                c_batch.append(c)
            yield np.array(a_batch),np.array(b_batch),np.array(c_batch)
class F1(ModelBasis,F1ModelArchitecture,DayDayNameF1IndexGenerator):
    test_days=10
    steps_per_epoch=10000
    def validation_steps(self):
        return int(len(store["Returns_scaled"].columns)*self.test_days/self.batch_size*3)

    def generate(self,test=False):
        store=self.store
        batch_size=self.batch_size
        H=self.stock_history_length
        r=store["Returns_scaled"].as_matrix()
        N,M=r.shape

        ibg = self.index_batch_generator(test)
        while True:
            batch_period_start_day,batch_prediction_day,batch_stock_number = next(ibg)
            X1=np.zeros((batch_size,H),np.float32)
            X2=np.zeros((batch_size,M),np.float32)
            Y=np.zeros((batch_size,1),np.float32)
            for i in range(self.batch_size):
                period_start_day=batch_period_start_day[i]
                prediction_day=batch_prediction_day[i]
                stock_number=batch_stock_number[i]
                X1[i,:]=r[period_start_day:period_start_day+H,stock_number]
                X2[i,:]=r[prediction_day,:]
                Y[i,0]=r[prediction_day,stock_number]
            yield [X1,X2],Y




class RE1ModelArchitecture:
    E=200
    def create_model(self):
        E=self.E
        logging.info("Create model %s (%d)"%(self.name(),E))
        store=self.store
        returns=store["Returns_scaled"]

        L=int(1.5*3*E)
        regularization=self.regularization

        stock_projection_inputs=Input(shape=(E,),name="stock_projection_inputs")
        compressed_returns_inputs=Input(shape=(E,),name="compressed_returns_inputs")
        mul=multiply([stock_projection_inputs, compressed_returns_inputs])
        fingerprint = concatenate([stock_projection_inputs,compressed_returns_inputs,mul])

        x=Dense(name="layer1",units=L, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(fingerprint)
        x=Dense(name="layer2",units=L, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(x)
        x=Dense(name="layer3",units=L, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(x)
        output=Dense(units=1, activation='linear', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(x)
        model = Model(inputs=[stock_projection_inputs,compressed_returns_inputs],outputs=output)
        model.compile(optimizer='rmsprop',loss='mse')
        modelfile = self.modelfile
        logging.info("Save model to %s"%modelfile)
        with open(modelfile,"w") as f:
            f.write(model.to_yaml())
        self.model = model
        return self

class CovE1ModelArchitecture:
    E=200
    def create_model(self):
        E=self.E
        logging.info("Create model %s (%d)"%(self.name(),E))
        store=self.store
        returns=store["Returns_scaled"]

        L=50
        regularization=self.regularization

        stockA_projection_inputs=Input(shape=(E,),name="stockA_projection_inputs")
        stockB_projection_inputs=Input(shape=(E,),name="stockB_projection_inputs")
        compressed_returns_inputs=Input(shape=(E,),name="compressed_returns_inputs")
        mulA=multiply([stockA_projection_inputs, compressed_returns_inputs])
        pseudo_returnsA=Dense(name="pseudo_returnsA",units=L, activation='linear', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(mulA)
        mulB=multiply([stockB_projection_inputs, compressed_returns_inputs])
        pseudo_returnsB=Dense(name="pseudo_returnsB",units=L, activation='linear', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(mulB)
        mulAB=multiply([pseudo_returnsA,pseudo_returnsB])
        
        fingerprint = concatenate([
            stockA_projection_inputs,
            stockB_projection_inputs,
            compressed_returns_inputs,
            mulA,
            mulB,
            mulAB
            ])

        x=Dense(name="layer1",units=6*E, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(fingerprint)
        x=Dense(name="layer2",units=3*E, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(x)
        x=Dense(name="layer3",units=1*E, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(x)
        output=Dense(units=1, activation='linear', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(x)
        model = Model(inputs=[stockA_projection_inputs,stockB_projection_inputs,compressed_returns_inputs],outputs=output)
        model.compile(optimizer='rmsprop',loss='mse')
        modelfile = self.modelfile
        logging.info("Save model to %s"%modelfile)
        with open(modelfile,"w") as f:
            f.write(model.to_yaml())
        self.model = model
        return self

class DayNameNameIndexGenerator2:
    def index_generator(self,test=False):
        r=store["Returns"]
        Nd=len(r.index)
        Ns=len(r.columns)
        test_days=self.test_days
        horizon=self.horizon
        if test:
            day_order=np.arange(Nd-test_days-horizon,Nd-horizon-1)
        else:
            day_order=np.arange(Nd-horizon-test_days-horizon-1)

        names_order1=np.arange(Ns)
        names_order2=np.arange(Ns)
        while True:            
            np.random.shuffle(day_order)
            np.random.shuffle(names_order1)
            np.random.shuffle(names_order2)
            for a,b,c in zip(day_order,names_order1,names_order2):
                if b==c:
                    yield a,b,c
                else:
                    yield a,b,c
                    yield a,c,b
                    yield a,b,b
                    yield a,c,c
    def index_batch_generator(self,test=False):
        g=self.index_generator(test)
        while True:
            a_batch=[]
            b_batch=[]
            c_batch=[]
            for i in range(self.batch_size):
                a,b,c=next(g)
                a_batch.append(a)
                b_batch.append(b)
                c_batch.append(c)
            yield np.array(a_batch),np.array(b_batch),np.array(c_batch)

class DayNameNameIndexGenerator2a:
    def index_generator(self,test=False):
        r=store["Returns"]
        Nd=len(r.index)
        Ns=len(r.columns)
        test_days=self.test_days
        horizon=self.horizon
        if test:
            day_order=np.arange(Nd-test_days-horizon,Nd-horizon-1)
        else:
            day_order=np.arange(Nd-horizon-test_days-horizon-1)

        names_order1=np.arange(Ns)
        names_order2=np.arange(Ns)
        while True:            
            np.random.shuffle(day_order)
            np.random.shuffle(names_order1)
            np.random.shuffle(names_order2)
            for a,b,c in zip(day_order,names_order1,names_order2):
                if b==c:
                    yield a,b,c
                    if a>0:
                        yield a-1,b,c
                else:
                    yield a,b,c
                    yield a,c,b
                    yield a,b,b
                    yield a,c,c
                    if a>0:
                        yield a-1,b,c
                        yield a-1,c,b
                        yield a-1,b,b
                        yield a-1,c,c
                        
    def index_batch_generator(self,test=False):
        g=self.index_generator(test)
        while True:
            a_batch=[]
            b_batch=[]
            c_batch=[]
            for i in range(self.batch_size):
                a,b,c=next(g)
                a_batch.append(a)
                b_batch.append(b)
                c_batch.append(c)
            yield np.array(a_batch),np.array(b_batch),np.array(c_batch)

class DayNameNameIndexGenerator2Sel:
    def index_generator(self,test=False):
        r=store["Returns"]
        selection_index=[list(r.columns).index(x) for x in self.selection]
        Nd=len(r.index)
        Ns=len(r.columns)
        test_days=self.test_days
        horizon=self.horizon
        if test:
            day_order=np.arange(Nd-test_days-horizon,Nd-horizon-1)
        else:
            day_order=np.arange(Nd-horizon-test_days-horizon-1)

        names_order=np.array(selection_index)
        while True:            
            np.random.shuffle(day_order)
            for a in day_order:
                for b in names_order:
                    for c in names_order:
                        yield a,b,c
                         
    def index_batch_generator(self,test=False):
        g=self.index_generator(test)
        while True:
            a_batch=[]
            b_batch=[]
            c_batch=[]
            for i in range(self.batch_size):
                a,b,c=next(g)
                a_batch.append(a)
                b_batch.append(b)
                c_batch.append(c)
            yield np.array(a_batch),np.array(b_batch),np.array(c_batch)

class DayNameNameIndexGenerator2SelHybrid:
    test_days=600
    
    def index_generator(self,test=False):
        r=store["Returns"]
        selection_index=[list(r.columns).index(x) for x in self.selection]
        Nd=len(r.index)
        Ns=len(r.columns)
        test_days=self.test_days
        horizon=self.horizon
        if test:
            day_order=np.arange(Nd-test_days-horizon-1,Nd-horizon-2)
        else:
            day_order=np.arange(Nd-horizon-test_days-horizon-2)

        names_order=np.array(selection_index)
        all_names_order1=np.arange(Ns)
        all_names_order2=np.arange(Ns)

        if test:
            while True:
                for a in day_order:
                    for b in names_order:
                        for c in names_order:
                            if b==c:
                                continue
                            yield a,b,c                
        else:
            while True:
                data=[]
                np.random.shuffle(day_order)
                np.random.shuffle(all_names_order1)
                np.random.shuffle(all_names_order2)
                for i in range(256):
                    if all_names_order1[i]==all_names_order2[i]:
                        continue
                    data.append([day_order[i],all_names_order1[i],all_names_order2[i]])
                    data.append([day_order[i],all_names_order2[i],all_names_order1[i]])
                for a in day_order:
                    for b in names_order:
                        for c in names_order:
                            if b==c:
                                continue
                            data.append([a,b,c])
                            data.append([a,c,b])
                                
                indices=np.array(data)
                np.random.shuffle(all_names_order2)
                for a,b,c in indices:
                    yield a,b,c
                         
    def index_batch_generator(self,test=False):
        g=self.index_generator(test)
        while True:
            a_batch=[]
            b_batch=[]
            c_batch=[]
            for i in range(self.batch_size):
                a,b,c=next(g)
                a_batch.append(a)
                b_batch.append(b)
                c_batch.append(c)
            yield np.array(a_batch),np.array(b_batch),np.array(c_batch)

class RE0ModelArchitecture:
    E=200
    def create_model(self):
        E=self.E
        logging.info("Create model %s (%d)"%(self.name(),E))
        store=self.store
        returns=store["Returns_scaled"]

        L=int(1.5*3*E)
        regularization=self.regularization

        stock_projection_inputs=Input(shape=(E,),name="stock_projection_inputs")
        compressed_returns_inputs=Input(shape=(E,),name="compressed_returns_inputs")
#        mul=multiply([stock_projection_inputs, compressed_returns_inputs])
        x=multiply([stock_projection_inputs, compressed_returns_inputs])
        #x=dot([stock_projection_inputs, compressed_returns_inputs],axes=1)
#        x = concatenate([stock_projection_inputs,compressed_returns_inputs,mul])
        output=Dense(units=1, activation='linear', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(x)
#        output=dot
        model = Model(inputs=[stock_projection_inputs,compressed_returns_inputs],outputs=output)
        model.compile(optimizer='rmsprop',loss='mse')
        modelfile = self.modelfile
        logging.info("Save model to %s"%modelfile)
        with open(modelfile,"w") as f:
            f.write(model.to_yaml())
        self.model = model
        return self

class DayNameIndexGenerator:
    def index_generator(self,test=False):
        r=store["Returns_scaled"].as_matrix()
        N,M=r.shape

        if test:
            day_order=N-np.arange(self.test_days)-1
        else:
            day_order=np.arange(N-self.test_days-2)
        N=len(day_order)

        stocks=np.arange(M)
        days_repeated=np.tile(day_order,M)
        stocks_repeated=np.repeat(stocks,N)
        o=np.concatenate((days_repeated,stocks_repeated)).reshape((2,-1)).T
        
        while True:
            np.random.shuffle(o)
            for x,y in o:
                yield x,y

    def index_batch_generator(self,test=False):
        g=self.index_generator(test)
        while True:
            a_batch=[]
            b_batch=[]
            for i in range(self.batch_size):
                a,b=next(g)
                a_batch.append(a)
                b_batch.append(b)
            yield np.array(a_batch),np.array(b_batch)

class RE1(ModelBasis,RE1ModelArchitecture,DayNameIndexGenerator):
    test_days=10
    steps_per_epoch=10000
    def validation_steps(self):
        return int(len(store["Returns_scaled"].columns)*self.test_days/self.batch_size)

    def generate(self,test=False):
        E=self.E
        store=self.store
        batch_size=self.batch_size
        r=store["Returns"].as_matrix()
        cr=store["CompressedReturns"].as_matrix()[:,:E]
        sp=store["StockProjections"].as_matrix()[:,:E]
        N,M=r.shape

        ibg = self.index_batch_generator(test)
        while True:
            batch_day,batch_stock_number = next(ibg)
            X1=sp[batch_stock_number]
            X2=cr[batch_day]
            Y=np.zeros((batch_size,1),np.float32)
            for i in range(batch_size):
                Y[i,0]=10*r[batch_day[i],batch_stock_number[i]]
            yield [X1,X2],Y
    def test1(self):
        E=self.E
        store=self.store
        batch_size=self.batch_size
        r=store["Returns"].as_matrix()
        cr=store["CompressedReturns"].as_matrix()[:,:E]
        sp=store["StockProjections"].as_matrix()[:,:E]
        N,M=r.shape

        batch_day=np.arange(128)
        batch_stock_number= np.zeros(128,np.int)
        batch_stock_number[:]=1
        X1=sp[batch_stock_number]
        X2=cr[batch_day]
        Y=np.zeros((batch_size,1),np.float32)
        for i in range(batch_size):
            Y[i,0]=10*r[batch_day[i],batch_stock_number[i]]
        Yp=self.model.predict([X1,X2], batch_size=batch_size)
        for y,yp in zip(Y,Yp):
            print(y[0],yp[0])

class RE0(ModelBasis,RE0ModelArchitecture,DayNameIndexGenerator):
    test_days=10
    steps_per_epoch=10000
    def validation_steps(self):
        return int(len(store["Returns_scaled"].columns)*self.test_days/self.batch_size)

    def generate(self,test=False):
        E=self.E
        store=self.store
        batch_size=self.batch_size
        r=store["ProjectedReturns"].as_matrix()
        cr=store["CompressedReturns"].as_matrix()[:,:E]
        sp=store["StockProjections"].as_matrix()[:,:E]
        N,M=r.shape

        ibg = self.index_batch_generator(test)
        while True:
            batch_day,batch_stock_number = next(ibg)
            X1=sp[batch_stock_number]
            X2=cr[batch_day]
            Y=np.zeros((batch_size,1),np.float32)
            for i in range(batch_size):
                Y[i,0]=10*r[batch_day[i],batch_stock_number[i]]
            yield [X1,X2],Y
    def test1(self):
        E=self.E
        store=self.store
        batch_size=self.batch_size
        r=store["ProjectedReturns"].as_matrix()
        cr=store["CompressedReturns"].as_matrix()[:,:E]
        sp=store["StockProjections"].as_matrix()[:,:E]
        N,M=r.shape

        batch_day=np.arange(128)
        batch_stock_number= np.zeros(128,np.int)
        batch_stock_number[:]=1

        X1=sp[batch_stock_number]
        X2=cr[batch_day]
        Y=np.zeros((batch_size,1),np.float32)
        for i in range(batch_size):
            Y[i,0]=10*r[batch_day[i],batch_stock_number[i]]
        Yp=self.model.predict([X1,X2], batch_size=batch_size)
        for y,yp in zip(Y,Yp):
            print(y[0],yp[0])

class CovE1Generator:
    test_days=10
    steps_per_epoch=10000
    horizon=10
    def validation_steps(self):
        return 500
        Ns=len(store["Returns"].columns)
        return int(Ns*Ns*self.test_days/self.batch_size)

    def generate(self,test=False):
        E=self.E
        store=self.store
        batch_size=self.batch_size
        r=store["Returns"].as_matrix()
        cr=store["CompressedReturns"].as_matrix()[:,:E]
        sp=store["StockProjections"].as_matrix()[:,:E]
        N,M=r.shape
        horizon=self.horizon
        ibg = self.index_batch_generator(test)
        while True:
            batch_day, batch_stockA_number, batch_stockB_number = next(ibg)
            X1=sp[batch_stockA_number]
            X2=sp[batch_stockB_number]
            X3=cr[batch_day]
            Y=np.zeros((batch_size,1),np.float32)
            for i in range(batch_size):
                rA=10*r[batch_day[i]:batch_day[i]+horizon,batch_stockA_number[i]]
                rB=10*r[batch_day[i]:batch_day[i]+horizon,batch_stockB_number[i]]
                Y[i,0]=np.average(rA*rB)
            yield [X1,X2,X3],Y
    def test1(self):
        pass

class CovE1fGenerator:
    test_days=10
    steps_per_epoch=10000
    horizon=10
    def validation_steps(self):
        return 500
        Ns=len(store["Returns"].columns)
        return int(Ns*Ns*self.test_days/self.batch_size)

    def generate(self,test=False):
        E=self.E
        store=self.store
        batch_size=self.batch_size
        r=store["Returns"].as_matrix()
        cr=store["CompressedReturns"].as_matrix()[:,:E]
        sp=store["StockProjections"].as_matrix()[:,:E]
        N,M=r.shape
        horizon=self.horizon
        ibg = self.index_batch_generator(test)
        while True:
            batch_day, batch_stockA_number, batch_stockB_number = next(ibg)
            X1=sp[batch_stockA_number]
            X2=sp[batch_stockB_number]
            X3=cr[batch_day]
            Y=np.zeros((batch_size,1),np.float32)
            for i in range(batch_size):
                rA=10*r[batch_day[i]+1:batch_day[i]+horizon,batch_stockA_number[i]]
                rB=10*r[batch_day[i]+1:batch_day[i]+horizon,batch_stockB_number[i]]
                Y[i,0]=np.average(rA*rB)
            yield [X1,X2,X3],Y
    def test1(self):
        pass

class CovE1fGenerator100:
    test_days=10
    steps_per_epoch=10000
    horizon=10
    def validation_steps(self):
        return 500
        Ns=len(store["Returns"].columns)
        return int(Ns*Ns*self.test_days/self.batch_size)

    def generate(self,test=False):
        E=self.E
        store=self.store
        batch_size=self.batch_size
        r=store["Returns"].as_matrix()
        cr=store["CompressedReturns"].as_matrix()[:,:E]
        sp=store["StockProjections"].as_matrix()[:,:E]
        N,M=r.shape
        horizon=self.horizon
        ibg = self.index_batch_generator(test)
        while True:
            batch_day, batch_stockA_number, batch_stockB_number = next(ibg)
            X1=sp[batch_stockA_number]
            X2=sp[batch_stockB_number]
            X3=cr[batch_day]
            Y=np.zeros((batch_size,1),np.float32)
            for i in range(batch_size):
                rA=100*r[batch_day[i]+1:batch_day[i]+horizon,batch_stockA_number[i]]
                rB=100*r[batch_day[i]+1:batch_day[i]+horizon,batch_stockB_number[i]]
                Y[i,0]=np.average(rA*rB)
            yield [X1,X2,X3],Y

class CovE1(ModelBasis,CovE1ModelArchitecture,DayNameNameIndexGenerator2a,CovE1Generator):
    test_days=10
    steps_per_epoch=10000
    horizon=10

class CovE1Sel(ModelBasis,CovE1ModelArchitecture,DayNameNameIndexGenerator2Sel,CovE1Generator):
    test_days=10
    horizon=10
    selection=["GOOGL","AAPL","NFLX","INTC"]
    steps_per_epoch=1024*len(selection)*len(selection)/256

class CovE1SelTMK(ModelBasis,CovE1ModelArchitecture,DayNameNameIndexGenerator2Sel,CovE1Generator):
    test_days=10
    horizon=10
    selection=["GOOGL","AAPL","NFLX","INTC","TMK"]
    steps_per_epoch=1024*len(selection)*len(selection)/256

class CovE1SelIBM(ModelBasis,CovE1ModelArchitecture,DayNameNameIndexGenerator2Sel,CovE1Generator):
    test_days=10
    horizon=10
    selection=["GOOGL","AAPL","NFLX","INTC","IBM"]
    steps_per_epoch=1024*len(selection)*len(selection)/256

class CovE1fSel6x6(ModelBasis,CovE1ModelArchitecture,DayNameNameIndexGenerator2Sel,CovE1fGenerator):
    test_days=10
    horizon=10
    selection=["GOOGL","AAPL","NFLX","INTC","IBM","MSFT"]
    steps_per_epoch=1024*len(selection)*len(selection)/256

class CovE1SelMSFT(ModelBasis,CovE1ModelArchitecture,DayNameNameIndexGenerator2Sel,CovE1Generator):
    test_days=10
    horizon=10
    selection=["GOOGL","AAPL","NFLX","INTC","IBM","MSFT"]
    steps_per_epoch=1024*len(selection)*len(selection)/256
    def test1(self):
        E=self.E
        store=self.store
        batch_size=self.batch_size
        R=store["ProjectedReturns"]
        r=store["ProjectedReturns"].as_matrix()
        cr=store["CompressedReturns"].as_matrix()[:,:E]
        sp=store["StockProjections"].as_matrix()[:,:E]
        N,M=r.shape
        horizon=self.horizon

        selection_index=[list(R.columns).index(x) for x in self.selection]
        data=[]
        for i in range(len(r)-self.horizon):
#        for i in range(200):
            print(i)
            for a in selection_index:
                nameA=R.columns[a]
                for b in selection_index:
                    nameB=R.columns[b]
                    X1=sp[a].reshape((200,1))
                    X2=sp[b].reshape((200,1))
                    X3=cr[i].reshape((200,1))
                    Yab=self.model.predict([X1.T,X2.T,X3.T])[0,0]
                    Yaa=self.model.predict([X1.T,X1.T,X3.T])[0,0]
                    Ybb=self.model.predict([X2.T,X2.T,X3.T])[0,0]
                    rA=10*r[i:i+horizon,a]
                    rB=10*r[i:i+horizon,b]
                    sA=np.sqrt(np.average(rA*rA))
                    sB=np.sqrt(np.average(rB*rB))
                    Yt=np.average(rA*rB)
                    CorrP=Yab/np.sqrt(Yaa*Ybb)
                    CorrT=Yt/(sA*sB)
                    data.append(dict(date=R.index[i],A=nameA,B=nameB,Yp=Yab,Yt=Yt,CorrP=CorrP,CorrT=CorrT))
        df=pd.DataFrame(data,columns=["date","A","B","Yp","Yt","CorrP","CorrT"])
        df.to_csv("6x6.csv")

class CovE1Hybrid(ModelBasis,CovE1ModelArchitecture,DayNameNameIndexGenerator2SelHybrid,CovE1fGenerator):
    test_days=600
    selection=["GOOGL","AAPL","NFLX","INTC","IBM","MSFT"]
    steps_per_epoch=600*len(selection)*len(selection)/256+2
    multiplier=10
    def validation_steps(self):
        return self.steps_per_epoch
    def test1(self):
        E=self.E
        store=self.store
        batch_size=self.batch_size
        R=store["Returns"]
        r=store["Returns"].as_matrix()
        cr=store["CompressedReturns"].as_matrix()[:,:E]
        sp=store["StockProjections"].as_matrix()[:,:E]
        N,M=r.shape
        horizon=self.horizon

        selection_index=[list(R.columns).index(x) for x in self.selection]
        data=[]
        for i in range(len(r)-self.horizon):
#        for i in range(200):
            if not i%100:
                print(i)
            for a in selection_index:
                nameA=R.columns[a]
                for b in selection_index:
                    nameB=R.columns[b]
                    X1=sp[a].reshape((200,1))
                    X2=sp[b].reshape((200,1))
                    X3=cr[i].reshape((200,1))
                    Yab=self.model.predict([X1.T,X2.T,X3.T])[0,0]
                    Yaa=self.model.predict([X1.T,X1.T,X3.T])[0,0]
                    Ybb=self.model.predict([X2.T,X2.T,X3.T])[0,0]
                    rA=10*r[i+1:i+horizon+1,a]
                    rB=10*r[i+1:i+horizon+1,b]
                    sA=np.sqrt(np.average(rA*rA))
                    sB=np.sqrt(np.average(rB*rB))
                    Yt=np.average(rA*rB)
                    CorrP=Yab/np.sqrt(Yaa*Ybb)
                    CorrT=Yt/(sA*sB)
                    data.append(dict(date=R.index[i],A=nameA,B=nameB,Yp=Yab,Yt=Yt,CorrP=CorrP,CorrT=CorrT))
        df=pd.DataFrame(data,columns=["date","A","B","Yp","Yt","CorrP","CorrT"])
        df.to_csv("Hybrid_6x6.csv")

class CovE1Hybrid300(CovE1Hybrid):
    test_days=300
    
Process = eval(process)

p=Process(store, regularization=regularization,batch_size=batch, modelfile=modelfile, weightsfile=weightsfile)
#for i,g in zip(range(10),p.generate()):
#    pass
#exit(0)
#print (next(p.index_batch_generator()))
p.fit(epochs)
p.save_weights()

p.test1()

