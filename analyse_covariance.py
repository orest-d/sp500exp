import pandas as pd
import numpy as np

store = pd.HDFStore("logrc1.h5")

cov = store["Covariance"]
Cov=cov.as_matrix()
sigma = np.sqrt(Cov.diagonal())
Corr = Cov/np.outer(sigma,sigma)
cf = Corr.reshape((-1,))
index = np.argsort(cf)

data=[]
for i in index:
    a=i/len(Cov)
    b=i%len(Cov)
#    if a<b:
#        continue
    data.append(dict(A=a,B=b,StockA=cov.columns[a],StockB=cov.index[b],correlation=cf[i],covariance=Cov[a,b]))

df=pd.DataFrame(data)
df.to_csv("corrcov.csv")    