import pandas as pd
import logging
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation,ReLU,LeakyReLU
from keras.layers import Input, GaussianDropout, Dropout
from keras.models import Model
from keras.models import model_from_yaml
from keras import regularizers
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
#input()

store = pd.HDFStore("store.h5")
outputstore = pd.HDFStore("compress_output.h5")

cdf = store["Close"]

print(cdf.columns)

def make_model_seq(inputs,outputs,regularization=0.001):
    model = Sequential()
    L1 = int(2 * max(inputs, outputs))
    L2 = int(max(inputs/2,3*outputs))
    L3 = int(min(inputs/2,3*outputs))

    model.add(Dense(units=L1, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization),
                    input_dim=inputs))
    model.add(Dense(units=L2, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization),
                    input_dim=L1))
    model.add(Dense(units=L2, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization),
                    input_dim=L3))
    model.add(Dense(units=L3, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization),
                    input_dim=L3))
    model.add(
        Dense(units=outputs, activation='linear', use_bias=True, kernel_regularizer=regularizers.l2(regularization),
              input_dim=L3))
    model.add(
        Dense(units=L3, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization),
              input_dim=outputs))
    model.add(Dense(units=L3, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization),
                    input_dim=L3))
    model.add(Dense(units=L3, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization),
                    input_dim=L2))
    model.add(Dense(units=L1, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization),
                    input_dim=L2))
    model.add(Dense(units=inputs, activation='linear', use_bias=True, kernel_regularizer=regularizers.l2(regularization),
                    input_dim=L1))
    model.compile(optimizer='rmsprop', loss='mse')
#    logging.info("Save model to %s" % modelfile)
#    with open(modelfile, "w") as f:
#        f.write(model.to_yaml())
    return model

def make_model(inputs,outputs,regularization=0.0000001):
    L1 = int(2 * max(inputs, outputs))
    L2 = int(max(inputs/2,3*outputs))
    L3 = int(min(inputs/2,3*outputs))

    big_input = Input(shape=(inputs,))
    small_input = Input(shape=(outputs,))
    encoded=GaussianDropout(0.4)(big_input)
    encoded=Dense(L1,  use_bias=True, kernel_regularizer=regularizers.l2(regularization))(encoded)
    encoded=LeakyReLU()(encoded)
    encoded=GaussianDropout(0.3)(encoded)
#    encoded=Dense(L2, use_bias=True, kernel_regularizer=regularizers.l2(regularization))(encoded)
#    encoded=LeakyReLU()(encoded)
    encoded=Dense(L3, use_bias=True, kernel_regularizer=regularizers.l2(regularization))(encoded)
    encoded=LeakyReLU()(encoded)
    encoded=GaussianDropout(0.2)(encoded)
    encoded=Dense(outputs, activation='linear', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(encoded)

    decoded=Dense(L3, use_bias=True, kernel_regularizer=regularizers.l2(regularization*2))(encoded)
    decoded=LeakyReLU()(decoded)
#    decoded=Dense(L2, use_bias=True, kernel_regularizer=regularizers.l2(regularization*2))(decoded)
#    decoded=LeakyReLU()(decoded)
    decoded=Dense(L1, use_bias=True, kernel_regularizer=regularizers.l2(regularization))(decoded)
    decoded=LeakyReLU()(decoded)
    decoded=Dense(inputs, activation='linear', use_bias=True, kernel_regularizer=regularizers.l2(regularization*2))(decoded)

    autoencoder = Model(big_input, decoded)
    autoencoder.compile(optimizer='rmsprop', loss='mse')
    encoder = Model(big_input,encoded)

#    deco = autoencoder.layers[-4](small_input)
#    deco = autoencoder.layers[-3](deco)
#    deco = autoencoder.layers[-2](deco)
#    deco = autoencoder.layers[-1](deco)

 #   decoder = Model(small_input,deco)

    return autoencoder,encoder#,decoder


def make_model_par(inputs,outputs,L1f=1,L2f=1,L3f=1, dropout=0.4, activation="LeakyReLU()",regularization=0.0000001,optimizer='rmsprop',loss='mse'):
    L1 = int(L1f * max(inputs, outputs))
    L2 = int(L2f * max(inputs/2,3*outputs))
    L3 = int(L3f * min(inputs/2,3*outputs))

    activation = eval(activation)
    big_input = Input(shape=(inputs,))
    small_input = Input(shape=(outputs,))
    encoded=GaussianDropout(dropout)(big_input)
    encoded=Dense(L1,  use_bias=True, kernel_regularizer=regularizers.l2(regularization))(encoded)
    encoded=LeakyReLU()(encoded)
    encoded=GaussianDropout(dropout*0.75)(encoded)
#    encoded=Dense(L2, use_bias=True, kernel_regularizer=regularizers.l2(regularization))(encoded)
#    encoded=LeakyReLU()(encoded)
    encoded=Dense(L3, use_bias=True, kernel_regularizer=regularizers.l2(regularization))(encoded)
    encoded=LeakyReLU()(encoded)
    encoded=GaussianDropout(dropout*0.5)(encoded)
    encoded=Dense(outputs, activation='linear', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(encoded)

    decoded=Dense(L3, use_bias=True, kernel_regularizer=regularizers.l2(regularization*2))(encoded)
    decoded=LeakyReLU()(decoded)
#    decoded=Dense(L2, use_bias=True, kernel_regularizer=regularizers.l2(regularization*2))(decoded)
#    decoded=LeakyReLU()(decoded)
    decoded=Dense(L1, use_bias=True, kernel_regularizer=regularizers.l2(regularization*2))(decoded)
    decoded=LeakyReLU()(decoded)
    decoded=Dense(inputs, activation='linear', use_bias=True, kernel_regularizer=regularizers.l2(regularization*2))(decoded)

    autoencoder = Model(big_input, decoded)
    autoencoder.compile(optimizer=optimizer, loss=loss)
    encoder = Model(big_input,encoded)

#    deco = autoencoder.layers[-4](small_input)
#    deco = autoencoder.layers[-3](deco)
#    deco = autoencoder.layers[-2](deco)
#    deco = autoencoder.layers[-1](deco)

 #   decoder = Model(small_input,deco)

    return autoencoder,encoder#,decoder


def evaluate_model(eigenvectors,r,validation_r,n,epochs,L1f=1,L2f=1,L3f=1, dropout=0.4, activation="LeakyReLU()",regularization=0.0000001,optimizer='rmsprop',loss='mse'):
    inputs = r.shape[1]
    autoencoder, encoder = make_model_par(inputs, n, L1f=L1f,L2f=L2f,L3f=L3f, dropout=dropout, activation=activation,regularization=regularization,optimizer=optimizer,loss=loss)
    history = autoencoder.fit(r, r, batch_size=200, validation_data=(validation_r, validation_r), epochs=epochs,
                              verbose=0)
    print("  fit1")
    ar = autoencoder.predict(r)
    validation_ar = autoencoder.predict(validation_r)

    O = eigenvectors[:n].T
    y = np.dot(r, O)
    x = np.dot(y, O.T)
    validation_y = np.dot(validation_r, O)
    validation_x = np.dot(validation_y, O.T)

    std_r = (r - x).std()
    std_validation_r = (validation_r - validation_x).std()

    std_ar = (r - ar).std()
    std_validation_ar = (validation_r - validation_ar).std()

    return dict(
        epochs=epochs,
        n=n,
        L1f=L1f, L2f=L2f, L3f=L3f, dropout=dropout, activation=activation, regularization=regularization,
        optimizer=optimizer, loss=loss,
        std_r=std_r,
        std_validation_r=std_validation_r,
        std_ar=std_ar,
        std_validation_ar=std_validation_ar,
    )

def _logrc1(store,H=600):
    print("close")
    close=store["Close"].copy()
    close.fillna(method="ffill",inplace=True)
    close.fillna(0,inplace=True)

    print("returns")
    returns=(close/close.shift(-1)).apply(np.log)

    remove=returns.apply(np.isfinite).sum()<0.9*len(returns)
    print("Remove (less than 90% data in total):",returns.columns[remove],remove.sum())
    logging.warning("Remove (less than 90% data in total):"+str(returns.columns[remove])+" "+str(remove.sum()))
    returns=returns.loc[:,~remove]
    close=close.loc[:,~remove]

    remove=returns.apply(np.isfinite)[:H].sum()<0.9*H
    logging.warning("Remove (less than 90% data in covariance period):"+str(returns.columns[remove])+" "+str(remove.sum()))
    print("Remove (less than 90% data in the covariance period) :",returns.columns[remove],remove.sum())
    returns=returns.loc[:,~remove]
    close=close.loc[:,~remove]

    returns.fillna(0,inplace=True)
    returns[~returns.apply(np.isfinite)]=0
    returns=returns[1:-1]
    close=close[1:-1]


    full_r=returns.values
    logging.info("Returns shape (full):"+str(full_r.shape))
    r=returns.values[:H].T
    validation_r=returns.values[H:]
    logging.info("Returns shape (covariance period):"+str(r.shape))


    cov=pd.DataFrame(columns=returns.columns,index=returns.columns)
    for i,n in enumerate(returns.columns):
        c=(r[i]*r).mean(axis=1)
        cov.loc[n,:]=c
    C=np.array(cov.values,dtype=np.double)
    eigenvalues,eigenvectors=np.linalg.eigh(C)
    eigenvalues=np.flip(eigenvalues,0)
    eigenvectors=eigenvectors.T
    eigenvectors=np.flip(eigenvectors,0)
    eigen_df=pd.DataFrame(np.hstack((eigenvalues.reshape((-1,1)),eigenvectors)),columns=["Eigenvalue"]+list(cov.columns))
    cov.to_csv("tmp.csv")
    cov=pd.read_csv("tmp.csv",index_col=0)
    outputstore["Covariance"]=cov
    d=np.sqrt(cov.values.diagonal())
    corr = cov/np.outer(d,d)
    outputstore["Correlations"]=corr
    outputstore["Eigenvectors"]=eigen_df

    r=r.T

    inputs = r.shape[1]
    with open("dim_std.csv","w") as f:
#        f.write("i;std;std_full;dC;maxrelC;eigenvalue\n")
        f.write("i;std;std_full;std_validation;astd;astd_full;astd_validation;astd2;astd_full2;astd_validation2\n")

#        for i in range(1,len(eigenvectors)):

        for i in []:#range(20,100,10):
            print (i)
            autoencoder, encoder = make_model(inputs, i)
            history = autoencoder.fit(r, r, batch_size=200, validation_data=(validation_r, validation_r), epochs=2000, verbose=0)
            print ("  fit1")
            ar = autoencoder.predict(r)
            full_ar=autoencoder.predict(full_r)
            validation_ar=autoencoder.predict(validation_r)
            history = autoencoder.fit(r, r, batch_size=200, validation_data=(validation_r, validation_r), epochs=6000, verbose=0)
            print ("  fit2")
            ar2 = autoencoder.predict(r)
            full_ar2=autoencoder.predict(full_r)
            validation_ar2=autoencoder.predict(validation_r)

            O=eigenvectors[:i].T
            y=np.dot(r,O)
            x=np.dot(y,O.T)
            C1=np.dot(O,np.dot(np.diag(eigenvalues[:i]),O.T))
            dC=(C-C1).std()
            maxrelC=np.max(np.abs((C-C1)/C))

            full_y=np.dot(full_r,O)
            full_x=np.dot(full_y,O.T)

            validation_y=np.dot(validation_r,O)
            validation_x=np.dot(validation_y,O.T)

            std_r=(r-x).std()
            std_full_r=(full_r-full_x).std()
            std_validation_r=(validation_r-validation_x).std()

            std_ar=(r-ar).std()
            std_full_ar=(full_r-full_ar).std()
            std_validation_ar=(validation_r-validation_ar).std()

            std_ar2=(r-ar2).std()
            std_full_ar2=(full_r-full_ar2).std()
            std_validation_ar2=(validation_r-validation_ar2).std()

            #f.write("%3d;%+10.8f;%+10.8f;%+10.8f;%+10.8f;%+12.8f\n"%(i,(r-x).std(),(full_r-full_x).std(),dC,maxrelC,eigenvalues[i-1]))

            f.write("%(i)3d;%(std_r)+10.8f;%(std_full_r)+10.8f;%(std_validation_r)+10.8f;%(std_ar)+10.8f;%(std_full_ar)+10.8f;%(std_validation_ar)+10.8f;%(std_ar2)+10.8f;%(std_full_ar2)+10.8f;%(std_validation_ar2)+10.8f;\n"%locals())
            f.flush()

        data=[]
        for n in [20,50,100]:
            for optimizer in ["sgd","rmsprop","adam"]:
                for activation in ["LeakyReLU()","ReLU()"]:
                    for regularization in [0.00000001,0.0000001,0.0000002,0.000001]:
                        for dropout in [0.01, 0.1, 0.2, 0.4, 0.6]:
                            for L1f in [0.5,1,2]:
                                for L2f in [0.5,1, 2]:
                                    for L3f in [0.5,1, 2]:
                                        for epochs in [500,1000,5000]:
                                            d=evaluate_model(eigenvectors, r, validation_r, n, epochs, L1f=L1f, L2f=L2f, L3f=L3f, dropout=dropout,
                                                               activation=activation, regularization=regularization, optimizer=optimizer, loss='mse')
                                            data.append(d)
                                            print(d)
                                            df=pd.DataFrame(data)
                                            df.to_csv("compress_analysis.csv")
    print(r.shape)
    inputs = r.shape[1]
    outputs = 10
    autoencoder, encoder, decoder = make_model(inputs,outputs)
    print("r",r.shape)
    print("validation_r",validation_r.shape)
    print("full_r",full_r.shape)
    history=autoencoder.fit(r,r, validation_data=(validation_r,validation_r), epochs=20, verbose=1)
#    history=autoencoder.fit(r,r,epochs=20, verbose=1)


    #history = autoencoder.fit(full_r, full_r, validation_split=0.5, epochs=20, verbose=1)
#    history=model.fit(r,r, epochs=20, verbose=1)



_logrc1(store)