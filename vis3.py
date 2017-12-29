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
store1 = pd.HDFStore("store2.h5")



close=store["Close"].copy()
close.fillna(method="ffill",inplace=True)
close.fillna(0,inplace=True)

returns=(close-close.shift())/close
returns.fillna(0,inplace=True)
returns.columns=[c+"_Return" for c in returns.columns]

volume = store["Volume"].copy()
volume.fillna(method="ffill",inplace=True)
volume.fillna(0,inplace=True)
volume.columns=[c+"_Volume" for c in volume.columns]

X=close.join(returns).join(volume)
#for name in close.columns:
#    r   = returns.loc[:,name+"_Return"]
#    cov = returns.multiply(r,axis="rows")
#    cov.columns = ["%s_%s_Cov"%(name,c) for c in close.columns]
#    X=X.join(cov)

store1["X"]=X
x=X.as_matrix()

scaler = preprocessing.StandardScaler().fit(x)
x=scaler.transform(x)
store1["X_scaled"]=pd.DataFrame(x,columns=X.columns,index=X.index)


pca = PCA(n_components=3)
print (x.shape)
X_new=pca.fit_transform(x)
print (X_new.shape)
print(pca.explained_variance_ratio_)
print(pca.singular_values_) 

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


index=np.zeros(len(X_new),dtype=np.bool)
for i in range(len(X_new)):
    dv=X_new-X_new[i]
    dv=np.sqrt(np.sum(dv*dv,axis=1))
    index[i]=np.sum(dv<3)>3
    print(i,dv.shape)

trace1 = go.Scatter3d(x=X_new[:,0],y=X_new[:,1],z=X_new[:,2],mode="lines",text=X.index)
trace2 = go.Scatter3d(x=X_new[index,0],y=X_new[index,1],z=X_new[index,2],mode="markers",text=X.index)
fig = go.Figure(data=[trace1,trace2])
plot(fig, filename='vis3.html')

with open("pca3.scad","w") as f:
    f.write("""a=[
""")
    f.write(",\n".join("    [%f,%f,%f]"%tuple(v) for v in X_new[:]))
    f.write("];\n")

    f.write("""


 $fn=6;

module rod(a, b, r) {
    //translate(a) sphere(r=r);
    //translate(b) sphere(r=r);
    dir = b-a;
    h   = norm(dir);
    if(dir[0] == 0 && dir[1] == 0) {
        // no transformation necessary
        cylinder(r=r, h=h);
    }
    else {
        w  = dir / h;
        u0 = cross(w, [0,0,1]);
        u  = u0 / norm(u0);
        v0 = cross(w, u);
        v  = v0 / norm(v0);
        multmatrix(m=[[u[0], v[0], w[0], a[0]],
                      [u[1], v[1], w[1], a[1]],
                      [u[2], v[2], w[2], a[2]],
                      [0,    0,    0,    1]])
        cylinder(r=r, h=h);
    }
}
 
 union(){
   for (i=[0:len(a)]){
       translate(a[i])sphere(r=2.5);
   }
   for (i=[1:len(a)]){
       rod(a[i-1],a[i],1.5);
   }
 }
""")

    
