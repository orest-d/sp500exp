import pandas as pd
import numpy as np
import logging
from os.path import exists
from sklearn import preprocessing
from sklearn.decomposition import PCA


inputfile='all_stocks_5yr.csv' 
storefile='store.h5'
columns=['Open', 'High', 'Low', 'Close', 'Volume']

if not exists(storefile):
    logging.info("Load csv")
    allstock = pd.read_csv(inputfile)
    print (allstock.columns)


    store = pd.HDFStore(storefile)

    for column in columns:
        logging.info("process "+column)
        df = allstock.pivot(index='Date', columns='Name', values=column)
        store[column]=df
    store.close()

store = pd.HDFStore(storefile)
store1 = pd.HDFStore("store1.h5")



df=store["Close"]
#print(df[:5].AAL)
df.fillna(method="ffill",inplace=True)
df.fillna(0,inplace=True)
#print(df[:5].AAL)

df1=df-df.shift()
df1.fillna(0,inplace=True)
df.columns=[c+"_Close" for c in df.columns]
df1.columns=[c+"_Return" for c in df1.columns]
#print(df.columns)
#print(df1.columns)
df2=df.join(df1)
#print(df2.columns)
store1["X"]=df2
store1["Y"]=df1.shift(-1)
x=df2.as_matrix()

scaler = preprocessing.StandardScaler().fit(x)
x=scaler.transform(x)
#print(np.mean(x[:,0]),np.mean(x[0,:]),np.std(x[:,0]),np.std(x[0,:]))
store1["X_scaled"]=pd.DataFrame(x,columns=df2.columns,index=df2.index)

dfy=store1["Y"]
dfy.fillna(0,inplace=True)
y=dfy.as_matrix()
scaler = preprocessing.StandardScaler().fit(y)
y=scaler.transform(y)
store1["Y_scaled"]=pd.DataFrame(y,columns=dfy.columns,index=dfy.index)

store1.close()


store1 = pd.HDFStore("store1.h5")
X=store1["X_scaled"].as_matrix()
inputs=X.shape[1]
N=X.shape[0]
assert(len(X)==N)
Y=store1["Y_scaled"].as_matrix()
outputs=Y.shape[1]
assert(len(X)==len(Y))

pca = PCA(n_components=3)
print (X.shape)
X_new=pca.fit_transform(X)
print (X_new.shape)
print(pca.explained_variance_ratio_)
print(pca.singular_values_) 

pca2 = PCA(n_components=3)
print (X.shape)
X_new2=pca2.fit_transform(X.T)
print (X_new2.shape)
print(pca2.explained_variance_ratio_)
print(pca2.singular_values_) 

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

L=int(len(X_new2)/2)

trace1 = go.Scatter3d(x=X_new[:,0],y=X_new[:,1],z=X_new[:,2],mode="lines",text=df.index)
trace2 = go.Scatter3d(x=X_new2[:L,0],y=X_new2[:L,1],z=X_new2[:L,2])
trace3 = go.Scatter3d(x=X_new2[L:,0],y=X_new2[L:,1],z=X_new2[L:,2])
fig = go.Figure(data=[trace1,trace2,trace3])
plot(fig, filename='vis1.html')

index=np.zeros(len(X_new),dtype=np.bool)
for i in range(len(X_new)):
    dv=X_new-X_new[i]
    dv=np.sqrt(np.sum(dv*dv,axis=1))
    index[i]=np.sum(dv<2.5)>3
    print(i,dv.shape)

with open("pca.jscad","w") as f:
    f.write("""var a=[
""")
    f.write(",\n".join("    [%f,%f,%f]"%tuple(v) for v in X_new[index]))
    f.write("];\n")

    f.write("""
function main() {
  var u=[];
  for (i=0;i<a.length;i++){
      u.push(sphere({r:1,center:true}).translate(a[i]));
  }
   return union(u);
}
""")

with open("pca.scad","w") as f:
    f.write("""a=[
""")
    f.write(",\n".join("    [%f,%f,%f]"%tuple(v) for v in X_new[index]))
    f.write("];\n")

    f.write("""
 $fn=10;
 union(){
   for (i=[0:len(a)]){
       translate(a[i])sphere(r=2.5);
   }
 }
""")
 
 
