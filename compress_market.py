import pandas as pd
import logging
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Input
from keras.models import Model
from keras.models import model_from_yaml
from keras import regularizers

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
    L1 = int(1 * max(inputs, outputs))
    L2 = int(max(inputs/2,3*outputs))
    L3 = int(min(inputs/2,3*outputs))

    big_input = Input(shape=(inputs,))
    small_input = Input(shape=(outputs,))
    encoded=Dense(L1, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(big_input)
    encoded=Dense(L2, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(encoded)
#    encoded=Dense(L3, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(encoded)
    encoded=Dense(outputs, activation='linear', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(encoded)

    decoded=Dense(L3, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(encoded)
    decoded=Dense(L2, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(decoded)
 #   decoded=Dense(L1, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(decoded)
    decoded=Dense(inputs, activation='linear', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(decoded)

    autoencoder = Model(big_input, decoded)
    autoencoder.compile(optimizer='rmsprop', loss='mse')
    encoder = Model(big_input,encoded)

    deco = autoencoder.layers[-3](small_input)
#    deco = autoencoder.layers[-3](deco)
    deco = autoencoder.layers[-2](deco)
    deco = autoencoder.layers[-1](deco)

    decoder = Model(small_input,deco)

    return autoencoder,encoder,decoder

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

        for i in range(1,100):
            autoencoder, encoder, decoder = make_model(inputs, i)
            history = autoencoder.fit(r, r, validation_data=(validation_r, validation_r), epochs=500, verbose=1)
            ar = autoencoder.predict(r)
            full_ar=autoencoder.predict(full_r)
            validation_ar=autoencoder.predict(validation_r)
            history = autoencoder.fit(r, r, validation_data=(validation_r, validation_r), epochs=1000, verbose=1)
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