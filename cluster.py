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

from keras.models import Sequential,Model
from keras.layers import Dense, Activation, Input, concatenate, BatchNormalization, multiply,dot
from keras.models import model_from_yaml
from keras import regularizers


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
    kmeans = KMeans(n_clusters=20).fit(mm)

    df = pd.DataFrame(pca, index=days,columns=["c%d"%i for i in range(pca.shape[1])])
    print(kmeans.labels_)
    df.loc[:,"label"]=kmeans.labels_



    data = []
    labelcount=[]
    for label in sorted(df.label.unique()):
        index = df.label == label
        labelcount.append((sum(index),label))
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

    maxlabel = max(labelcount)[1]
    y = df.label==maxlabel
    df.loc[:,"y"]=y

    c = BinaryClassifier(mm.shape[1])
    x=m[:-1]
    y=y[1:]
    assert len(x) == len(y)
    c.fit(x,y,10)
    yp=c.predict(x)
    df.loc[1:,"yp"]=yp
    df.loc[1:,"yptrunc"]=yp>0.5

    print ("yptrunc",df.y.sum(),df.yptrunc.sum(),len(df.yptrunc))
    data = []
    for label in [True,False]:
        index = df.yptrunc == label
        trace = go.Scatter3d(

            x=df.c0[index],
            y=df.c1[index],
            z=df.c2[index],
            mode='markers',
            name = str(label)
        )
        data.append(trace)
    fig = go.Figure(data=data)
    plot(fig, filename=outputfile+'_y.html')


    output["diag"] = df

class Measure:
    def __init__(self, returns):
        self.m = returns*returns
    def fraction(self, selection):
        return float(min(np.sum(selection), np.sum(~selection))) / len(selection)
    def __call__(self,selection):
        f = self.fraction(selection)

        m1 = np.mean(self.m[selection],axis=0)
        m2 = np.mean(self.m[~selection],axis=0)
        d = m1-m2
        return f*f*np.sqrt(np.sum(d*d))

class BinaryClassifier:
    def __init__(self,inputs):
        self.inputs = inputs
        L = int(1.5 * inputs)
        regularization =0.0
        input_layer=Input(shape=(inputs,),name="inputs")
        x=Dense(name="layer1",units=L, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(input_layer)
        x=Dense(name="layer2",units=int(L/2), activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(regularization))(x)
        output_layer=Dense(name="output", units=1, activation='sigmoid')(x)
        self.model = Model(inputs=input_layer,outputs=output_layer)
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    def fit(self, x,y, epochs=1):
        self.model.fit(x=x,y=y, epochs=epochs, verbose=1, shuffle=True)

    def predict(self,x):
        return self.model.predict(x)
    def save_weights(self,weightsfile):
        self.model.save_weights(weightsfile)
    def load_weights(self,weightsfile):
        self.model.load_weights(weightsfile, by_name=True)


class Genome:
    Model = BinaryClassifier
    def __init__(self,genome_size, identifier=None):
        if type(genome_size) is int:
            self.genome = np.random.rand(genome_size)>0.5
        else:
            self.genome = np.array(genome_size)
        self.identifier = identifier
        self.fitness = None
        self.model = self.Model(genome_size)
        if identifier is None:
            self.weightsfile = None
        else:
            self.weightsfile = "%s_weights.h5"%identifier
        self.predicted_genome = None

    def fit(self,x):
        self.model.fit(x,self.genome,epochs=10)
        self.save_weights()
        self.predicted_genome = self.model.predict(x)

    def save_weights(self,weightsfile=None):
        if weightsfile is None:
            weightsfile = self.weightsfile
        if weightsfile is not None:
            self.model.save_weights(weightsfile)
    def load_weights(self,weightsfile):
        if weightsfile is None:
            weightsfile = self.weightsfile
        if weightsfile is not None:
            if exists(weightsfile):
                logging.info("Loading genome weights %s"%weightsfile)
                self.model.load_weights(weightsfile, by_name=True)
            else:
                logging.info("Genome weights %s do not exist"%weightsfile)
        else:
            logging.info("Genome %s weightfile not specified" % self.identifier)

    def __len__(self):
        return len(self.genome)

    def clone(self,identifier=None):
        g=Genome(self.genome.copy(), identifier)
        g.load_weights(self.weightsfile)
        g.save_weights()
        return g

    def mutate(self, identifier=None):
        g=self.clone(identifier)
        n = int(np.random.uniform(0,len(g)))
        g.genome[n] = not g.genome[n]
        return g

    def predicted(self, identifier=None):
        if self.predicted_genome is not None:
            if np.any(self.predicted_genome!=self.genome):
                g=self.clone(identifier)
                g.genome = self.predicted_genome.copy()
                g.predicted_genome = self.predicted_genome.copy()
                return g

    def cross(self,parent,identifier=None):
        g=self.clone(identifier)
        index = np.random.randint(0,len(self),len(self)/2)
        g.genome[index] = parent.genome[index]
        return g

    def __str__(self):
        return "Genome(%d,'%s', sum=%d, fitness=%s)"%(len(self.genome), self.identifier,np.sum(self.genome),str(fitness))



class GeneticOptimizer:
    def __init__(self,pool_size,x):
        self.pool_size = pool_size
        self.genome_size = len(x)
        self.x=x
        self.pool = []
        self.measure = Measure(x)
    def step(self):
        while len(self.pool)<self.pool_size:
            identifier = len(self.pool)+1
            g = Genome(self.genome_size,identifier=identifier)
        g.fit(x)
        self.pool.append((self.measure(g.predicted_genome), g))
        action = np.random.randint(1,5)
        self.pool = sorted(self.pool)
        out = self.pool[0]
        self.pool = self.pool[1]
        if action == 1:
            a = np.random.randint(0,len(self.pool))
            b = np.random.randint(0,len(self.pool))
            if a!=b:
                parent_a = self.pool[a][1]
                parent_b = self.pool[b][1]
                g=parent_a.cross(parent_b,identifier=out.identifier)
                g.fit(x)
                self.pool.append((self.measure(g.predicted_genome),g))
        if action == 2:
            i = np.random.randint(0,len(self.pool))
            g = self.pool[i][1].mutate(out.identifier)
            g.fit(x)
            self.pool.append((self.measure(g.predicted_genome), g))
        if action == 3:
            i = np.random.randint(0,len(self.pool))
            g = self.pool[i][1].predicted(out.identifier)
            if g is not None:
                self.pool.append((self.measure(g.predicted_genome), g))
        if action == 4:
            i = np.random.randint(0,len(self.pool))
            g = self.pool[i][1].fit(self.x)
            self.pool[i] = (self.measure(g.predicted_genome), g)
            
store = pd.HDFStore(inputfile)
table = store[table_name]

output = pd.HDFStore(outputfile+".h5")
process = eval(process_function)
process(table,output)
output.close()