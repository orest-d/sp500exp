import pandas as pd
import numpy as np
import logging
from os.path import exists
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

def scale_df(df,subtract_mean=True):
    scale = pd.DataFrame(data=[dict(df.mean()),dict(df.std()),dict(df.mean())],columns=df.columns,index=["mean","std","true_mean"])
    if not subtract_mean:
        scale.loc["mean",:]=0
        
    scale.loc["mean"].fillna(0,inplace=True)
    scale.loc["std"].fillna(1,inplace=True)
    scale.loc["true_mean"].fillna(0,inplace=True)
    v=scale.loc["std"].as_matrix()
    v[v==0]=1
    scale.loc["std"]=v
    return scale

def trivial_scale_df(df):
    return pd.DataFrame(data={c:(0,1) for c in df.columns},columns=df.columns,index=["mean","std"])

def rescale_df(df,scale):
    return (df-scale.loc["mean"])/scale.loc["std"]

def inverse_scale_df(df,scale):
    return df*scale.loc["std"]+scale.loc["mean"]

def make_output(output,X,Y,test,scale):
    assert(len(X)==len(Y))
    std = X.std().as_matrix()
    X=X.loc[:,(std!=0) & np.isfinite(std)]

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

def sigma10a(store):
    print("close")
    close=store["Close"].copy()
    close.fillna(method="ffill",inplace=True)
    close.fillna(0,inplace=True)

    print("returns")
    returns=(close-close.shift(1))/close
    returns.fillna(0,inplace=True)
    returns.columns=[c+"_Return" for c in returns.columns]

#    print("volume")
#    volume = store["Volume"].copy()
#    volume.fillna(method="ffill",inplace=True)
#    volume.fillna(0,inplace=True)
#    volume.columns=[c+"_Volume" for c in volume.columns]

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

 #   X=close.join(returns).join(volume).join(sigma)
    X=close.join(returns).join(sigma)
    X=X[N:-N]
    Y=sigma.shift(-int(N/2))[:-N]
    return X,Y

def sigma20f(store):
    print("close")
    close=store["Close"].copy()
    close.fillna(method="ffill",inplace=True)
    close.fillna(0,inplace=True)

    print("returns")
    returns=(close-close.shift(1))/close
    returns.fillna(0,inplace=True)
    returns.columns=[c+"_Return" for c in returns.columns]

#    print("volume")
#    volume = store["Volume"].copy()
#    volume.fillna(method="ffill",inplace=True)
#    volume.fillna(0,inplace=True)
#    volume.columns=[c+"_Volume" for c in volume.columns]

    print("sigma")
    N=20
    lag=10
    r=returns.as_matrix()
    columns=[c+"_Sigma%d"%N for c in close.columns]
    index=close.index[N:]
    data_a = np.zeros((len(index),len(columns)),np.double)
    data_f = np.zeros((len(index),len(columns)),np.double)

    for i in range(len(r)-N-lag):
        a=r[i:i+N,:]
        b=r[i+lag:i+N+lag,:]
        sigma_a=np.sqrt(np.average(a*a,axis=0))
        sigma_b=np.sqrt(np.average(b*b,axis=0))
        f=sigma_b/sigma_a
        f[~np.isfinite(f)]=1
        data_a[i]=sigma_a
        data_f[i]=f
    sigma  = pd.DataFrame(data_a,columns=columns,index=index)
    sigmaf = pd.DataFrame(data_f,columns=columns,index=index)
    sigma.fillna(0,inplace=True)
    sigmaf.fillna(1,inplace=True)

#    X=close.join(returns).join(volume).join(sigma)
    X=close.join(returns).join(sigma)
    X=X[N:-N-lag]
    Y=sigmaf[N:-lag]
    return X,Y

def min10(store):
    print("close")
    close=store["Close"].copy()
    close.fillna(method="ffill",inplace=True)
    close.fillna(0,inplace=True)

#    print("high")
#    high=store["High"].copy()
#    high.fillna(method="ffill",inplace=True)
#    high.fillna(0,inplace=True)
#    high.columns=[c+"_High" for c in close.columns]
#
#    print("low")
#    low=store["Low"].copy()
#    low.fillna(method="ffill",inplace=True)
#    low.fillna(0,inplace=True)
#    low.columns=[c+"_Low" for c in close.columns]

    print("returns")
    returns=(close-close.shift(1))/close
    returns.fillna(0,inplace=True)
    returns.columns=[c+"_Return" for c in returns.columns]

    print("returns1")
    returns1=(close.shift(1)-close.shift(2))/close.shift(1)
    returns1.fillna(0,inplace=True)
    returns1.columns=[c+"_Return1" for c in returns.columns]

    print("min10")
    N=10
    r=returns.as_matrix()
    columns=[c+"_Min%d"%N for c in close.columns]
    index=close.index[N:]
    data = np.zeros((len(index),len(columns)),np.double)

    for i in range(len(r)-N):
        a=r[i:i+N,:]
        data[i]=np.min(a,axis=0)
    d = pd.DataFrame(data,columns=columns,index=index)
    d.fillna(0,inplace=True)

#    X=close.join(high).join(low).join(returns).join(d)
    X=close.join(returns).join(returns1).join(d)
    X=X[N:-N]
    Y=d.shift(-N)[:-N]
    return X,Y

def small(store):
    print("close")
    stock=["INTC"]
    close=store["Close"].loc[:,stock].copy()
    close.fillna(method="ffill",inplace=True)
    close.fillna(0,inplace=True)

    print("returns")
    returns=(close-close.shift(1))/close
    returns.fillna(0,inplace=True)
    returns.columns=[c+"_Return" for c in returns.columns]

    N=20        
    r=returns.as_matrix()
    columns=[c+"_%d"%N for c in close.columns]
    index=close.index[N:]
    data = np.zeros((len(index),len(columns)),np.double)
    X=close
    for i in range(len(r)-N):
        a=r[i:i+N,:]
        data[i]=np.sqrt(np.sum(a*a,axis=0)/N)
    for i in range(N):
        print("returns %02d"%i)
        returns_i=(close.shift(i)-close.shift(i+1))/close.shift(i)
        returns_i.fillna(0,inplace=True)
        returns_i.columns=["%s_Return%02d"%(c,i) for c in close.columns]
        X=X.join(returns_i)
    d = pd.DataFrame(data,columns=columns,index=index)
    d.fillna(0,inplace=True)

    X=X[N:]
    Y=d
    return X,Y

def small2(store):
    print("close")
    stock=["AAPL","INTC"]
    stock=["INTC"]
    close=store["Close"].loc[:,stock].copy()
    close.fillna(method="ffill",inplace=True)
    close.fillna(0,inplace=True)

    print("returns")
    returns=(close-close.shift(1))/close
    returns.fillna(0,inplace=True)
    returns.columns=[c+"_Return" for c in returns.columns]

    N=20        
    r=returns.as_matrix()
    columns=[c+"_%d"%N for c in close.columns]
    index=close.index[N:]
    data = np.zeros((len(index),len(columns)),np.double)
    X=close
    for i in range(len(r)-N):
        a=r[i:i+N,:]
        data[i]=np.sqrt(np.sum(a*a,axis=0)/N)
    for i in range(1,N+1):
        print("Close %02d"%i)
        close_i=close.shift(i)
        close_i.fillna(0,inplace=True)
        close_i.columns=["%s_Close%02d"%(c,i) for c in close.columns]
        X=X.join(close_i)
    d = pd.DataFrame(data,columns=columns,index=index)
    d.fillna(0,inplace=True)

    X=X[N:]
    Y=d
    return X,Y

def name_to_pca_2(df1,df2,n_observations=100,n_components=20):
    from sklearn import preprocessing
    from sklearn.decomposition import PCA
    df1=df1.fillna(method="ffill",inplace=False)
    df1.fillna(0,inplace=True)
    df2=df2.fillna(method="ffill",inplace=False)
    df2.fillna(0,inplace=True)
    names = df1.columns
    assert(all(n1==n2 for n1,n2 in zip(df1.columns,df2.columns)))

    m1=df1.as_matrix()
    m1=m1[:n_observations]
    m2=df1.as_matrix()
    m2=m2[:n_observations]
    
    scaler1 = preprocessing.StandardScaler().fit(m1)
    m1=scaler1.transform(m1)
    scaler2 = preprocessing.StandardScaler().fit(m2)
    m2=scaler2.transform(m2)
    m=np.concatenate((m1,m2))

    pca = PCA(n_components=n_components)
    m_pca=pca.fit_transform(m.T)
    print("explained variance ratio %s"%pca.explained_variance_ratio_)
#    print("singular values          %s"%pca.singular_values_)
    print (m_pca.shape)#,pca.inverse_transform(m_pca[0]))
#    for a,b in zip(list(pca.inverse_transform(m_pca[0])),list(m[:,0])):
#        print (a,b)
    maxdev=0
    for i in range(len(names)):
        a=m[:,i]
        b=pca.inverse_transform(m_pca[i])
        delta = b-a
        std=np.std(delta)
#        print(i,np.mean(delta),std)
        maxdev=max(maxdev,std)
    print("maxdev %s"%maxdev)
    return names,m_pca
    
def name_to_pca(df,n_observations=100,n_components=20):
    from sklearn import preprocessing
    from sklearn.decomposition import PCA
    df=df.fillna(method="ffill",inplace=False)
    df.fillna(0,inplace=True)
    names = df.columns
    m=df.as_matrix()
    m=m[:n_observations]
    scaler = preprocessing.StandardScaler().fit(m)
    m=scaler.transform(m)
    print(np.mean(m[:,0]),np.mean(m[0,:]),np.std(m[:,0]),np.std(m[0,:]))
    print(m.shape)
    pca = PCA(n_components=n_components)
    m_pca=pca.fit_transform(m.T)
    print("explained variance ratio %s"%pca.explained_variance_ratio_)
#    print("singular values          %s"%pca.singular_values_)
    print (m_pca.shape)#,pca.inverse_transform(m_pca[0]))
#    for a,b in zip(list(pca.inverse_transform(m_pca[0])),list(m[:,0])):
#        print (a,b)
    maxdev=0
    for i in range(len(names)):
        a=m[:,i]
        b=pca.inverse_transform(m_pca[i])
        delta = b-a
        std=np.std(delta)
#        print(i,np.mean(delta),std)
        maxdev=max(maxdev,std)
    print("maxdev %s"%maxdev)
    return names,m_pca

def day_to_pca(df,n_observations=500,n_components=20):
    from sklearn import preprocessing
    from sklearn.decomposition import PCA
    df=df.fillna(method="ffill",inplace=False)
    df.fillna(0,inplace=True)
    names = df.columns
    days=df.index
    m=df.as_matrix()
    n_observations = len(m) if n_observations is None else n_observations
    m=m[:n_observations]
    m=m.T
    scaler = preprocessing.StandardScaler().fit(m)
    m=scaler.transform(m)
    print(np.mean(m[:,0]),np.mean(m[0,:]),np.std(m[:,0]),np.std(m[0,:]))
    print(m.shape)
    pca = PCA(n_components=n_components)
    m_pca=pca.fit_transform(m.T)
    print("explained variance ratio %s"%pca.explained_variance_ratio_)
#    print("singular values          %s"%pca.singular_values_)
    print (m_pca.shape)#,pca.inverse_transform(m_pca[0]))
#    for a,b in zip(list(pca.inverse_transform(m_pca[0])),list(m[:,0])):
#        print (a,b)
    maxdev=0
    m=df.as_matrix()
    m_pca=pca.transform(m)
    print (m_pca.shape)#,pca.inverse_transform(m_pca[0]))
    for i in range(n_observations):
        a=m[i,:]
        b=pca.inverse_transform(m_pca[i])
        delta = b-a
        std=np.std(delta)
        #print(i,np.mean(delta),std)
        maxdev=max(maxdev,std)
    print("maxdev        %s"%maxdev)
    for i in range(len(days)):
        a=m[i,:]
        b=pca.inverse_transform(m_pca[i])
        delta = b-a
        std=np.std(delta)
#        print(i,np.mean(delta),std)
        maxdev=max(maxdev,std)
    print("maxdev (full) %s"%maxdev)
    return days,m_pca

def icorr(store,n_components=20):
    print("close")
    close=store["Close"].copy()
    close.fillna(method="ffill",inplace=True)
    close.fillna(0,inplace=True)

    print("returns")
    returns=(close-close.shift(1))/close
    returns.fillna(0,inplace=True)
    returns.columns=[c+"" for c in returns.columns]
    returns=returns[1:-1]
    r=returns.as_matrix()

    rnames,rpca=name_to_pca(returns,n_observations=100,n_components=n_components)
    a_columns=["a%03d"%(i) for i in range(n_components)]
    b_columns=["b%03d"%(i) for i in range(n_components)]
    columns=["return"]+a_columns+b_columns
    Y_columns=["result"]
    X=pd.DataFrame(columns=columns)
    Y=pd.DataFrame(columns=Y_columns)
    index=0
    for i,aname in enumerate(rnames):
#        if aname not in ["INTC","AAPL"]:
#            continue
        print (i,aname)
        for j,bname in enumerate(rnames):
#            if bname not in ["INTC","AAPL"]:
#                continue
            print (i,j,aname,bname)
            df=pd.DataFrame(columns=columns,index=range(index,index+len(r)))
            df.loc[:,"return"]=returns[aname].as_matrix()
            df.loc[:,a_columns]=rpca[i]
            df.loc[:,b_columns]=rpca[j]
            X=X.append(df)
            df=pd.DataFrame(r[:,i]*r[:,j],columns=Y_columns,index=range(index,index+len(r)))
            Y=Y.append(df)
            index+=len(r)
    return X,Y

def rpcasigma1(store,n_components=25):
    print("close")
    close=store["Close"].copy()
    close.fillna(method="ffill",inplace=True)
    close.fillna(0,inplace=True)

    print("returns")
    returns=(close-close.shift(1))/close
    returns.fillna(0,inplace=True)
    returns.columns=[c+"" for c in returns.columns]
    returns=returns[1:-1]
    sigma=returns.std()
    r=returns.as_matrix()
    
    rnames,rpca=name_to_pca(returns,n_observations=200,n_components=n_components)
    a_columns=["rpca%03d"%(i) for i in range(n_components)]
    columns=a_columns
    Y_columns=["sigma"]
    X=pd.DataFrame(columns=columns)
    Y=pd.DataFrame(columns=Y_columns)
    for i,aname in enumerate(rnames):
        X.loc[i,a_columns]=np.array(rpca[i],dtype=np.double)
        Y.loc[i,"sigma"]=float(sigma[aname])

    X.to_csv("tmp.csv")
    X=pd.read_csv("tmp.csv",index_col=0)
    Y.to_csv("tmp.csv")
    Y=pd.read_csv("tmp.csv",index_col=0)
    return X,Y

def rpcasigma_m(store,n_components=20):
    print("close")
    close=store["Close"].copy()
    close.fillna(method="ffill",inplace=True)
    close.fillna(0,inplace=True)

    print("returns")
    returns=(close-close.shift(1))/close
    returns.fillna(0,inplace=True)
    returns.columns=[c+"" for c in returns.columns]
    returns=returns[1:-1]
    r=returns.as_matrix()
    
    days,pca=day_to_pca(returns,n_observations=500,n_components=n_components)
    a_columns=["Mpca%03d"%(i) for i in range(n_components)]
    columns=a_columns
    Y_columns=["Msigma"]
    X=pd.DataFrame(columns=columns)
    Y=pd.DataFrame(columns=Y_columns)
    for i,day in enumerate(days):
        X.loc[i,a_columns]=np.array(pca[i],dtype=np.double)
        Y.loc[i,"Msigma"]=returns.loc[day].as_matrix().std()

    X.to_csv("tmp.csv")
    X=pd.read_csv("tmp.csv",index_col=0)
    Y.to_csv("tmp.csv")
    Y=pd.read_csv("tmp.csv",index_col=0)
    return X,Y

def rpcar_m(store,n_components=20):
    print("close")
    close=store["Close"].copy()
    close.fillna(method="ffill",inplace=True)
    close.fillna(0,inplace=True)

    print("returns")
    returns=(close-close.shift(1))/close
    returns.fillna(0,inplace=True)
    returns.columns=[c+"" for c in returns.columns]
    returns=returns[1:-1]
    r=returns.as_matrix()
    
    days,pca=day_to_pca(returns,n_observations=500,n_components=n_components)
    a_columns=["Mpca%03d"%(i) for i in range(n_components)]
    columns=a_columns
    Y_columns=returns.columns
    X=pd.DataFrame(columns=columns)
    Y=pd.DataFrame(columns=Y_columns)
    for i,day in enumerate(days):
        if day not in returns.index:
            continue
        X.loc[i,a_columns]=np.array(pca[i],dtype=np.double)
        Y.loc[i,Y_columns]=returns.loc[day]

    X.to_csv("tmp.csv")
    X=pd.read_csv("tmp.csv",index_col=0)
    Y.to_csv("tmp.csv")
    Y=pd.read_csv("tmp.csv",index_col=0)
    return X,Y

def cpcasigma1(store,n_components=25):
    print("close")
    close=store["Close"].copy()
    close.fillna(method="ffill",inplace=True)
    close.fillna(0,inplace=True)

    print("returns")
    returns=(close-close.shift(1))/close
    returns.fillna(0,inplace=True)
    returns.columns=[c+"" for c in returns.columns]
    returns=returns[1:-1]
    sigma=returns.std()
    r=returns.as_matrix()
    
    names,cpca=name_to_pca(close,n_observations=200,n_components=n_components)
    a_columns=["cpca%03d"%(i) for i in range(n_components)]
    columns=a_columns
    Y_columns=["sigma"]
    X=pd.DataFrame(columns=columns)
    Y=pd.DataFrame(columns=Y_columns)
    for i,aname in enumerate(names):
        X.loc[i,a_columns]=np.array(cpca[i],dtype=np.double)
        Y.loc[i,"sigma"]=float(sigma[aname])

    X.to_csv("tmp.csv")
    X=pd.read_csv("tmp.csv",index_col=0)
    Y.to_csv("tmp.csv")
    Y=pd.read_csv("tmp.csv",index_col=0)
    return X,Y

def rc2pcasigma1(store,c_components=20,r_components=20):
    return rcpcasigma1(store,c_components=c_components,r_components=r_components)
def rc3pcasigma1(store,c_components=25,r_components=15):
    return rcpcasigma1(store,c_components=c_components,r_components=r_components)
def rc4pcasigma1(store,c_components=26,r_components=24):
    return rcpcasigma1(store,c_components=c_components,r_components=r_components)
def c50pcasigma1(store):
    return cpcasigma1(store,n_components=50)
def r50pcasigma1(store):
    return rpcasigma1(store,n_components=50)

def rcpcasigma1(store,c_components=25,r_components=25):
    print("close")
    close=store["Close"].copy()
    close.fillna(method="ffill",inplace=True)
    close.fillna(0,inplace=True)

    print("returns")
    returns=(close-close.shift(1))/close
    returns.fillna(0,inplace=True)
    returns.columns=[c+"" for c in returns.columns]
    returns=returns[1:-1]
    sigma=returns.std()
    r=returns.as_matrix()
    
    names,cpca=name_to_pca(close,n_observations=200,n_components=c_components)
    names,rpca=name_to_pca(returns,n_observations=200,n_components=r_components)
    cpca_columns=["cpca%03d"%(i) for i in range(c_components)]
    rpca_columns=["rpca%03d"%(i) for i in range(r_components)]
    columns=cpca_columns+rpca_columns
    Y_columns=["sigma"]
    X=pd.DataFrame(columns=columns)
    Y=pd.DataFrame(columns=Y_columns)
    for i,aname in enumerate(names):
        X.loc[i,cpca_columns]=np.array(cpca[i],dtype=np.double)
        X.loc[i,rpca_columns]=np.array(rpca[i],dtype=np.double)
        Y.loc[i,"sigma"]=float(sigma[aname])

    X.to_csv("tmp.csv")
    X=pd.read_csv("tmp.csv",index_col=0)
    Y.to_csv("tmp.csv")
    Y=pd.read_csv("tmp.csv",index_col=0)
    return X,Y

def rcpca50sigma_m(store):
    return rcpcasigma_m(store,c_components=50,r_components=50)
def rcpca40sigma_m(store):
    return rcpcasigma_m(store,c_components=40,r_components=40)
def rcpca30sigma_m(store):
    return rcpcasigma_m(store,c_components=30,r_components=30)
def rcpca20sigma_m(store):
    return rcpcasigma_m(store,c_components=20,r_components=20)

def rcpcasigma_m(store,c_components=50,r_components=50):
    print("close")
    close=store["Close"].copy()
    close.fillna(method="ffill",inplace=True)
    close.fillna(0,inplace=True)

    print("returns")
    returns=(close-close.shift(1))/close
    returns.fillna(0,inplace=True)
    returns.columns=[c+"" for c in returns.columns]
    returns=returns[1:-1]
    close=close[1:-1]
    r=returns.as_matrix()
    
    cdays,cpca=day_to_pca(close,n_observations=500,n_components=c_components)
    rdays,rpca=day_to_pca(returns,n_observations=500,n_components=r_components)
    assert(len(cdays)==len(rdays))
    assert(all(x==y for x,y in zip(cdays,rdays)))
    cpca_columns=["cpcaM%03d"%(i) for i in range(c_components)]
    rpca_columns=["rpcaM%03d"%(i) for i in range(r_components)]
    columns=cpca_columns+rpca_columns
    Y_columns=["sigmaM"]
    X=pd.DataFrame(columns=columns)
    Y=pd.DataFrame(columns=Y_columns)
    for i,day in enumerate(rdays):
        X.loc[day,cpca_columns]=np.array(cpca[i],dtype=np.double)
        X.loc[day,rpca_columns]=np.array(rpca[i],dtype=np.double)
        Y.loc[i,"sigmaM"]=returns.loc[day].as_matrix().std()

    X.to_csv("tmp.csv")
    X=pd.read_csv("tmp.csv",index_col=0)
    Y.to_csv("tmp.csv")
    Y=pd.read_csv("tmp.csv",index_col=0)
    return X,Y

def crcpcasigma1(store,n_components=50):
    print("close")
    close=store["Close"].copy()
    close.fillna(method="ffill",inplace=True)
    close.fillna(0,inplace=True)

    print("returns")
    returns=(close-close.shift(1))/close
    returns.fillna(0,inplace=True)
    returns.columns=[c+"" for c in returns.columns]
    returns=returns[1:-1]
    sigma=returns.std()
    r=returns.as_matrix()
    
    names,pca=name_to_pca_2(close,returns,n_observations=200,n_components=n_components)
    a_columns=["crcpca%03d"%(i) for i in range(n_components)]
    columns=a_columns
    Y_columns=["sigma"]
    X=pd.DataFrame(columns=columns)
    Y=pd.DataFrame(columns=Y_columns)
    for i,aname in enumerate(names):
        X.loc[i,a_columns]=np.array(pca[i],dtype=np.double)
        Y.loc[i,"sigma"]=float(sigma[aname])

    X.to_csv("tmp.csv")
    X=pd.read_csv("tmp.csv",index_col=0)
    Y.to_csv("tmp.csv")
    Y=pd.read_csv("tmp.csv",index_col=0)
    return X,Y

def pcas_return(store):
    print("close")
    close=store["Close"].copy()
    close.fillna(method="ffill",inplace=True)
    close.fillna(0,inplace=True)

    print("returns")
    returns=(close-close.shift(1))/close
    returns.fillna(0,inplace=True)
    returns.columns=[c+"" for c in returns.columns]
    returns=returns[1:-1]
    close=close[1:-1]
    r=returns.as_matrix()
    
    market_components=40
    stock_components=25
    cdays,mcpca=day_to_pca(close,n_observations=500,n_components=market_components)
    rdays,mrpca=day_to_pca(returns,n_observations=500,n_components=market_components)

    cnames,cpca=name_to_pca(close,n_observations=500,n_components=stock_components)
    rnames,rpca=name_to_pca(returns,n_observations=500,n_components=stock_components)

    assert(len(cdays)==len(rdays))
    assert(all(x==y for x,y in zip(cdays,rdays)))
    assert(len(cnames)==len(rnames))
    assert(all(x==y for x,y in zip(cnames,rnames)))
    mcpca_columns=["cpcaM%03d"%(i) for i in range(market_components)]
    mrpca_columns=["rpcaM%03d"%(i) for i in range(market_components)]
    cpca_columns=["cpca%03d"%(i) for i in range(stock_components)]
    rpca_columns=["rpca%03d"%(i) for i in range(stock_components)]
    columns=mcpca_columns+mrpca_columns+cpca_columns+rpca_columns
    Y_columns=["return"]
    X=pd.DataFrame(columns=columns)
    Y=pd.DataFrame(columns=Y_columns)
    index=0
    for i,day in enumerate(rdays[:3]):
        print (i,day)
        for j,name in enumerate(rnames):
            X.loc[index,mcpca_columns]=np.array(mcpca[i],dtype=np.double)
            X.loc[index,mrpca_columns]=np.array(mrpca[i],dtype=np.double)
            X.loc[index,cpca_columns]=np.array(cpca[i],dtype=np.double)
            X.loc[index,rpca_columns]=np.array(rpca[i],dtype=np.double)
            Y.loc[index,"return"]=returns.loc[day,name]
            index+=1
    X.to_csv("tmp.csv")
    X=pd.read_csv("tmp.csv",index_col=0)
    Y.to_csv("tmp.csv")
    Y=pd.read_csv("tmp.csv",index_col=0)
    return X,Y

def _rc(store,outputstore):
    print("close")
    close=store["Close"].copy()
    close.fillna(method="ffill",inplace=True)
    close.fillna(0,inplace=True)

    print("returns")
    returns=(close-close.shift(1))/close
    returns.fillna(0,inplace=True)
    returns.columns=[c+"" for c in returns.columns]
    returns=returns[1:-1]
    close=close[1:-1]
    
    r=returns.as_matrix().T
    cov=pd.DataFrame(columns=returns.columns,index=returns.columns)
    for i,n in enumerate(returns.columns):
        c=(r[i]*r).mean(axis=1)
        cov.loc[n,:]=c
    cov.to_csv("tmp.csv")
    cov=pd.read_csv("tmp.csv",index_col=0)
    outputstore["Covariance"]=cov
    
    market_components=40
    stock_components=25
    cdays,mcpca=day_to_pca(close,n_observations=500,n_components=market_components)
    rdays,mrpca=day_to_pca(returns,n_observations=500,n_components=market_components)

    cnames,cpca=name_to_pca(close,n_observations=500,n_components=stock_components)
    rnames,rpca=name_to_pca(returns,n_observations=500,n_components=stock_components)

    assert(len(cdays)==len(rdays))
    assert(all(x==y for x,y in zip(cdays,rdays)))
    assert(len(cnames)==len(rnames))
    assert(all(x==y for x,y in zip(cnames,rnames)))
    assert(len(close.index)==len(cdays))
    assert(all(x==y for x,y in zip(close.index,cdays)))
    assert(len(returns.index)==len(cdays))
    assert(all(x==y for x,y in zip(returns.index,cdays)))
    assert(len(close.columns)==len(cnames))
    assert(all(x==y for x,y in zip(close.columns,cnames)))
    assert(len(returns.columns)==len(cnames))
    assert(all(x==y for x,y in zip(returns.columns,cnames)))

    mcpca_columns=["cpcaM%03d"%(i) for i in range(market_components)]
    mrpca_columns=["rpcaM%03d"%(i) for i in range(market_components)]
    cpca_columns=["cpca%03d"%(i) for i in range(stock_components)]
    rpca_columns=["rpca%03d"%(i) for i in range(stock_components)]

    days_df  = pd.DataFrame(np.concatenate((mcpca,mrpca),axis=1),
                            columns=mcpca_columns+mrpca_columns,
                            index=cdays)
    names_df = pd.DataFrame(np.concatenate((cpca,rpca),axis=1),
                            columns=cpca_columns+rpca_columns,
                            index=cnames)

    for name,df in [("Close",close),("Returns",returns),("Days",days_df),("Names",names_df)]:
        print("Df:"+name)
        outputstore[name] = df

        if scale.lower()=="yes":
            print ("Scale "+name)
            df_scale = scale_df(df)
        else:
            print ("Trivial Scale "+name)
            df_scale = trivial_scale_df(df)

        df_scaled                       = rescale_df(df,df_scale)
        df_scale.to_csv("x_scale.csv")
        output["%s_scaled"%name]        = df_scaled
        output["%s_scale"%name]         = df_scale

def _logrc(store,outputstore):
    print("close")
    close=store["Close"].copy()
    close.fillna(method="ffill",inplace=True)
    close.fillna(0,inplace=True)

    print("returns")    
    returns=(close/close.shift(-1)).apply(np.log)
    remove=returns.apply(np.isfinite).sum()<0.9*len(returns)
    print("Remove:",returns.columns[remove],remove.sum())
    returns.fillna(0,inplace=True)
    returns[~returns.apply(np.isfinite)]=0
    returns=returns[1:-1]
    close=close[1:-1]
    returns=returns.loc[:,~remove]
    close=close.loc[:,~remove]
    
    r=returns.as_matrix().T
    cov=pd.DataFrame(columns=returns.columns,index=returns.columns)
    for i,n in enumerate(returns.columns):
        c=(r[i]*r).mean(axis=1)
        cov.loc[n,:]=c
    cov.to_csv("tmp.csv")
    cov=pd.read_csv("tmp.csv",index_col=0)
    outputstore["Covariance"]=cov
    d=np.sqrt(cov.as_matrix().diagonal())
    corr = cov/np.outer(d,d)
    outputstore["Correlations"]=corr

    for name,df in [("Close",close),("Returns",returns)]:
        print("Df:"+name)
        outputstore[name] = df

        if scale.lower()=="yes":
            print ("Scale "+name)
            df_scale = scale_df(df)
        else:
            print ("Trivial Scale "+name)
            df_scale = trivial_scale_df(df)

        df_scaled                       = rescale_df(df,df_scale)
        df_scale.to_csv("x_scale.csv")
        output["%s_scaled"%name]        = df_scaled
        output["%s_scale"%name]         = df_scale

def _logrc1(store,outputstore,H=600,N=200):
    print("close")
    close=store["Close"].copy()
    close.fillna(method="ffill",inplace=True)
    close.fillna(0,inplace=True)

    print("returns")    
    returns=(close/close.shift(-1)).apply(np.log)
    H = (H+len(returns)-2)%(len(returns)-2)

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


    full_r=returns.as_matrix()
    logging.info("Returns shape (full):"+str(full_r.shape))
    r=returns.as_matrix()[:H].T
    logging.info("Returns shape (covariance period):"+str(r.shape))


    cov=pd.DataFrame(columns=returns.columns,index=returns.columns)
    for i,n in enumerate(returns.columns):
        c=(r[i]*r).mean(axis=1)
        cov.loc[n,:]=c
    C=np.array(cov.as_matrix(),dtype=np.double)
    eigenvalues,eigenvectors=np.linalg.eigh(C)
    eigenvalues=np.flip(eigenvalues,0)
    eigenvectors=eigenvectors.T
    eigenvectors=np.flip(eigenvectors,0)
    eigen_df=pd.DataFrame(np.hstack((eigenvalues.reshape((-1,1)),eigenvectors)),columns=["Eigenvalue"]+list(cov.columns))
    cov.to_csv("tmp.csv")
    cov=pd.read_csv("tmp.csv",index_col=0)
    outputstore["Covariance"]=cov
    d=np.sqrt(cov.as_matrix().diagonal())
    corr = cov/np.outer(d,d)
    outputstore["Correlations"]=corr
    outputstore["Eigenvectors"]=eigen_df

    r=r.T
    with open("dim_std.csv","w") as f:
        f.write("i;std;std_full;dC;maxrelC;eigenvalue\n")
        for i in range(1,len(eigenvectors)):
            O=eigenvectors[:i].T
            y=np.dot(r,O)
            x=np.dot(y,O.T)
            C1=np.dot(O,np.dot(np.diag(eigenvalues[:i]),O.T))
            dC=(C-C1).std()
            maxrelC=np.max(np.abs((C-C1)/C))
            full_y=np.dot(full_r,O)
            full_x=np.dot(full_y,O.T)
            f.write("%3d;%+10.8f;%+10.8f;%+10.8f;%+10.8f;%+12.8f\n"%(i,(r-x).std(),(full_r-full_x).std(),dC,maxrelC,eigenvalues[i-1]))

    O=eigenvectors[:N].T
    #inv_sqrt_L=np.diag(np.power(eigenvalues[:N],-0.5))
    #D=np.dot(O,inv_sqrt_L)
    y=np.dot(full_r,O)
    pr = np.dot(y,O.T)
    creturns_df = pd.DataFrame(y,index=returns.index,columns=["m%02d"%(i) for i in range(y.shape[1])])
    projected_returns_df = pd.DataFrame(pr,index=returns.index,columns=returns.columns)
    stock_df = pd.DataFrame(O,columns=["o%02d"%(i) for i in range(N)],index=returns.columns)

    for name,df,subtract_mean in [
        ("Close",close,True),
        ("Returns",returns,False),
        ("CompressedReturns",creturns_df,False),
        ("ProjectedReturns",projected_returns_df,False),
        ("StockProjections",stock_df,False)
        ]:
        print("Df:   "+name)
        outputstore[name] = df

        if scale.lower()=="yes":
            print ("Scale "+name)
            df_scale = scale_df(df,subtract_mean=subtract_mean)
        else:
            print ("Trivial Scale "+name)
            df_scale = trivial_scale_df(df)

        df_scaled                       = rescale_df(df,df_scale)
        df_scale.to_csv("x_scale.csv")
        output["%s_scaled"%name]        = df_scaled
        output["%s_scale"%name]         = df_scale
    
store = pd.HDFStore(storefile)
output = pd.HDFStore(outputfile)
logging.info("Process "+process_function)
process = eval(process_function)

if process_function[0]=="_":
    r=process(store,output)
    if r is not None:
        X,Y=r
        logging.info("Output "+process_function)
        make_output(output,X,Y,test,scale)
        
else:
    X,Y=process(store)
    logging.info("Output "+process_function)
    make_output(output,X,Y,test,scale)

store.close()
output.close()
logging.info("Finished")
