import pandas as pd
import numpy as np
import logging
from os.path import exists
import argparse
import sys
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description='Cluster data.')
parser.add_argument('-i','--input',    help='Inputfile .h5',default="cluster_input.h5")
parser.add_argument('-x','--index',    help='Index column (Date)',default="Date")
parser.add_argument('-o','--output',   help='Output  (clusters)',default="clusters")
parser.add_argument('-p','--process',  help='Process type',default="c1")
parser.add_argument('-t','--table',    help='Input table',default="Returns_scaled")

args = parser.parse_args()
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, filename='log.txt')

inputfile=args.input
indexcolumn = args.index
outputfile = args.output
process_function = args.process
table_name = args.table

logging.info("Input:        "+inputfile)
logging.info("Index column: "+indexcolumn)
logging.info("Output:       "+outputfile)
logging.info("Process:      "+process_function)
logging.info("Table:        "+table_name)


def day_to_pca(df,n_observations=None,n_components=3):
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

def c1(table,output):
    days,pca = day_to_pca(table)
    kmeans = KMeans(n_clusters=4, random_state=0).fit(table.as_matrix())

    df = pd.DataFrame(pca, index=days,columns=["c%d"%i for i in range(pca.shape[1])])
    print(kmeans.labels_)
    df.loc[:,"label"]=kmeans.labels_

    output["pca"] = df

    data = []
    for label in sorted(df.label.unique()):
        index = df.label == label
        trace = go.Scatter3d(

            x=df.c0[index],
            y=df.c1[index],
            z=df.c2[index],
            mode='markers',
            name = str(label)
        )
        data.append(trace)
    fig = go.Figure(data=data)
    plot(fig, filename=outputfile+'.html')

def c2(table,output):
    m = table.as_matrix()
    mm = m*m
    #table = pd.DataFrame(mm, index = table.index,columns=table.columns)
    days,pca = day_to_pca(table)
    kmeans = KMeans(n_clusters=10).fit(mm)

    df = pd.DataFrame(pca, index=days,columns=["c%d"%i for i in range(pca.shape[1])])
    print(kmeans.labels_)
    df.loc[:,"label"]=kmeans.labels_

    output["pca"] = df

    data = []
    for label in sorted(df.label.unique()):
        index = df.label == label
        trace = go.Scatter3d(

            x=df.c0[index],
            y=df.c1[index],
            z=df.c2[index],
            mode='markers',
            name = str(label)
        )
        data.append(trace)
    fig = go.Figure(data=data)
    plot(fig, filename=outputfile+'.html')

store = pd.HDFStore(inputfile)
table = store[table_name]

output = pd.HDFStore(outputfile+".h5")
process = eval(process_function)
process(table,output)
output.close()