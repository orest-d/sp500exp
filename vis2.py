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
df2=df.join(df1)

store1["X"]=df2
store1["Y"]=df1.shift(-1)
X=df2.as_matrix()

scaler = preprocessing.StandardScaler().fit(X)
X=scaler.transform(X)

inputs=X.shape[1]
N=X.shape[0]
assert(len(X)==N)

pca = PCA(n_components=3)
print (X.shape)
X_new=pca.fit_transform(X)
print (X_new.shape)
print(pca.explained_variance_ratio_)
print(pca.singular_values_) 

def blob(A,threshold,ior,rgbt,radius):
    spheres="\n".join("  sphere { <%f,%f,%f>, 5, %f }"%(v[0],v[1],v[2],radius) for v in A)
    s="""
blob {
  threshold %(threshold)s
  %(spheres)s
  scale 0.03
  pigment{ color rgbt < %(rgbt)s > }
  interior { ior %(ior)s }
  finish{ambient 1}  
  rotate <265+clock,5*clock,0>
}    
"""%locals()
    return s

with open("pov/SP500.pov","w") as f:
    f.write("""
#declare CamLook = <0,0,3>; // Camera's Look_at 
#declare CamLoc = <0,0.5,-6>; //where the camera's location is
#declare cam_z = 1.5; //the amount of camera zoom you want
#declare back_dist = 200; // how far away the background is
#declare cam_a = 4/3; // camera aspect ratio
#declare cam_s = <0,1,0>; // camera sky vectoy
#declare cam_d = vnormalize(CamLook-CamLoc); // camera direction vector
#declare cam_r = vnormalize(vcross(cam_s,cam_d)); // camera right vector
#declare cam_u = vnormalize(vcross(cam_d,cam_r)); // camera up vector
#declare cam_dir = cam_d * cam_z; // direction vector scaled
#declare cam_right = cam_r * cam_a; // right vector scaled

#declare fz = vlength(cam_dir);
#declare fx = vlength(cam_right)/2;
#declare fy = vlength(cam_u)/2; 

#macro OrientZ(p1,p2,cs)
  #local nz = vnormalize(p2-p1);
  #local nx = vnormalize(vcross(cs,nz)); 
  #local ny = vcross(nz,nx);
  matrix <nx.x,nx.y,nx.z, ny.x,ny.y,ny.z, nz.x,nz.y,nz.z, p1.x,p1.y,p1.z>          
#end

camera {
  location CamLoc
  up cam_u
  right cam_r * cam_a
  direction (cam_d * cam_z) 
}

box { <0,0,0> <1,1,0.1>
      pigment { image_map { png "background.png" 
                map_type 0 
                interpolate 2 } }
      finish { ambient 0.5 }
      translate <-0.5,-0.5,0>
      scale 2*<fx,fy,0.5>
      translate fz*z
      scale back_dist
      OrientZ(CamLoc,CamLook,cam_s) }



global_settings{
  ambient_light color rgb < 0.200000, 0.200000, 0.200000 >
}

background{color rgb < 0.000000, 0.000000, 0.000000 >}

light_source{
  < -30.000000, 30.000000, -30.000000 >, rgb < 1.000000, 1.000000, 1.000000 > shadowless
}
    
    """)
    #f.write(blob(X_new,0.1,1.001,"0,0,0.4,0.2"))
    #f.write(blob(X_new,0.5,1.01,"0,0,0.8,0.8"))
    f.write(blob(X_new,0.5,1.001,"0.0,0.0,0.5,0.05",0.5))
    f.write(blob(X_new,0.3,1.005,"0.1,0.1,0.8,0.8",1.0))
    f.write(blob(X_new,0.1,1.001,"0.3,0.3,0.9,0.8",2.5))
