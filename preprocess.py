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

args = parser.parse_args()
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, filename='log.txt')

inputfile=args.input
storefile=args.store
indexcolumn = args.index
namecolumn = args.name
outputfile = args.output
process = args.process
test = int(args.test)

logging.info("Input:        "+inputfile)
logging.info("Storage:      "+storefile)
logging.info("Index column: "+indexcolumn)
logging.info("Name column:  "+namecolumn)
logging.info("Output:       "+outputfile)
logging.info("Process:      "+process)
logging.info("Test:         "+str(test))

if not exists(storefile):
    logging.info("Load csv")
    input_df = pd.read_csv(inputfile)

    store = pd.HDFStore(storefile)

    for column in df.columns:
        if column in [indexcolumn,namecolumn]:
            continue
        logging.info("process "+column)
        df = input_df.pivot(index=indexcolumn, columns=namecolumn, values=column)
        store[column]=df
    store.close()

def returns(store,output,test):
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
    output["X"]=X
    output["Y"]=Y

    x=X.as_matrix()
    boundary=len(X)-test
    x_training=x[:boundary]
    x_test=x[boundary:]
    scaler = preprocessing.StandardScaler().fit(x_training)
    x_training=scaler.transform(x_training)
    x_test=scaler.transform(x_test)
    output["X_training"]=pd.DataFrame(x_training,columns=X.columns,index=X.index[:boundary])
    output["X_test"]=pd.DataFrame(x_test,columns=X.columns,index=X.index[boundary:])

    y=Y.as_matrix()
    y_training=y[:boundary]
    y_test=y[boundary:]
    scaler = preprocessing.StandardScaler().fit(y_training)
    y_training=scaler.transform(y_training)
    y_test=scaler.transform(y_test)
    output["Y_training"]=pd.DataFrame(y_training,columns=Y.columns,index=Y.index[:boundary])
    output["Y_test"]=pd.DataFrame(y_test,columns=Y.columns,index=Y.index[boundary:])


store = pd.HDFStore(storefile)
output = pd.HDFStore(outputfile)
process = eval(args.process)
process(store,output,test)
store.close()
output.close()
