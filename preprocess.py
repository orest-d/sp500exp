import pandas as pd
import numpy as np
import logging
from os.path import exists
from sklearn import preprocessing
import argparse
import sys

parser = argparse.ArgumentParser(description='Preprocess data.')
parser.add_argument('-i','--input',    help='Inputfile .csv (all_stocks_5yr.csv)',default="all_stocks_5yr.csv")
parser.add_argument('-s','--store',    help='Storage - h5 version of input, (store.h5)',default="store.h5")
parser.add_argument('-x','--index',    help='Index column (Date)',default="Date")
parser.add_argument('-n','--name',     help='Column-name column (Name)',default="Name")
parser.add_argument('-o','--output',   help='Output  (train.h5)',default="train.h5")
parser.add_argument('-p','--process',  help='Process type (returns)',default="returns")
parser.add_argument('-t','--test',     help='Number of samples in test (100)',default="100")
parser.add_argument('-c','--scale',    help='Scale yes/no (yes)',default="yes")

args = parser.parse_args()
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, filename='log.txt')

inputfile=args.input
storefile=args.store
indexcolumn = args.index
namecolumn = args.name
outputfile = args.output
process_function = args.process
test = int(args.test)
scale = args.scale

logging.info("Input:        "+inputfile)
logging.info("Storage:      "+storefile)
logging.info("Index column: "+indexcolumn)
logging.info("Name column:  "+namecolumn)
logging.info("Output:       "+outputfile)
logging.info("Process:      "+process_function)
logging.info("Test:         "+str(test))
logging.info("Scale:        "+scale)

if not exists(storefile):
    logging.info("Load csv")
    input_df = pd.read_csv(inputfile)

    store = pd.HDFStore(storefile)

    for column in input_df.columns:
        if column in [indexcolumn,namecolumn]:
            continue
        logging.info("process "+column)
        df = input_df.pivot(index=indexcolumn, columns=namecolumn, values=column)
        store[column]=df
    store.close()

def scale_df(df):
    return pd.DataFrame(data=[dict(df.mean()),dict(df.std())],columns=df.columns,index=["mean","std"])

def trivial_scale_df(df):
    return pd.DataFrame(data={c:(0,1) for c in df.columns},columns=df.columns,index=["mean","std"])

def rescale_df(df,scale):
    return (df-scale.loc["mean"])/scale.loc["std"]

def inverse_scale_df(df,scale):
    return df*scale.loc["std"]+scale.loc["mean"]

def make_output(output,X,Y,test,scale):
    boundary=len(X)-test

    x_training=X[:boundary]
    x_test=X[boundary:]

    y_training=Y[:boundary]
    y_test=Y[boundary:]

    if scale.lower()=="yes":
        print ("Scale")
        x_scale = scale_df(x_training)
        y_scale = scale_df(y_training)
    else:
        print ("Trivial Scale")
        x_scale = trivial_scale_df(x_training)
        y_scale = trivial_scale_df(y_training)

    x_training                     = rescale_df(x_training,x_scale)
    x_test                         = rescale_df(x_test,    x_scale)
    x_scale.loc["training_mean",:] = x_training.mean()
    x_scale.loc["training_std",:]  = x_training.std()
    x_scale.loc["test_mean",:]     = x_test.mean()
    x_scale.loc["test_std",:]      = x_test.std()
    x_scale.to_csv("x_scale.csv")
    output["X"]                    = X
    output["X_scale"]              = x_scale
    output["X_training"]           = x_training
    output["X_test"]               = x_test

    y_training                     = rescale_df(y_training,y_scale)
    y_test                         = rescale_df(y_test,    y_scale)
    y_scale.loc["training_mean",:] = y_training.mean()
    y_scale.loc["training_std",:]  = y_training.std()
    y_scale.loc["test_mean",:]     = y_test.mean()
    y_scale.loc["test_std",:]      = y_test.std()
    y_scale.to_csv("y_scale.csv")
    output["Y"]                    = Y
    output["Y_scale"]              = y_scale
    output["Y_training"]           = y_training
    output["Y_test"]               = y_test


def returns(store):
    print("close")
    close=store["Close"].copy()
    close.fillna(method="ffill",inplace=True)
    close.fillna(0,inplace=True)

    print("returns")
    returns=(close-close.shift())/close
    returns.fillna(0,inplace=True)
    returns.columns=[c+"_Return" for c in returns.columns]

    print("volume")
    volume = store["Volume"].copy()
    volume.fillna(method="ffill",inplace=True)
    volume.fillna(0,inplace=True)
    volume.columns=[c+"_Volume" for c in volume.columns]

    X=close.join(returns).join(volume)
    X=X[1:-1]
    Y=returns.shift(-1)[1:-1]
    return X,Y

def sigma10(store):
    print("close")
    close=store["Close"].copy()
    close.fillna(method="ffill",inplace=True)
    close.fillna(0,inplace=True)

    print("returns")
    returns=(close-close.shift(1))/close
    returns.fillna(0,inplace=True)
    returns.columns=[c+"_Return" for c in returns.columns]

    print("volume")
    volume = store["Volume"].copy()
    volume.fillna(method="ffill",inplace=True)
    volume.fillna(0,inplace=True)
    volume.columns=[c+"_Volume" for c in volume.columns]

    print("sigma")
    N=10
    r=returns.as_matrix()
    columns=[c+"_Sigma%d"%N for c in close.columns]
    index=close.index[N:]
    data = np.zeros((len(index),len(columns)),np.double)

    for i in range(len(r)-N):
        a=r[i:i+N,:]
        data[i]=np.sqrt(np.sum(a*a,axis=0)/N)
    sigma = pd.DataFrame(data,columns=columns,index=index)

    X=close.join(returns).join(volume).join(sigma)
    X=X[N:-N]
    Y=sigma.shift(-N)[:-N]
    return X,Y

store = pd.HDFStore(storefile)
output = pd.HDFStore(outputfile)
logging.info("Process "+process_function)
process = eval(process_function)
X,Y=process(store)
logging.info("Output "+process_function)
make_output(output,X,Y,test,scale)
store.close()
output.close()
logging.info("Finished")
