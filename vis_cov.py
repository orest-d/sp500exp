import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Hybrid_200_800.csv")


for A in ["GOOGL","AAPL","NFLX","INTC","IBM","MSFT"]:
    for B in ["GOOGL","AAPL","NFLX","INTC","IBM","MSFT"]:
        fig, ax = plt.subplots(nrows=2, ncols=1)
        index = (df.A==A) & (df.B==B)
        ax[0].plot(df[index].i,df[index].Yt,"y-",label="Observed (10d)")
        ax[0].plot(df[index].i,df[index].Yp,"b-",label="Prediction")
        ax[0].axvline(200)
        ax[0].axvline(800)
        #ax.plot(df[index].i,df[index].Yt,label="training")
        ax[0].legend()
        ax[0].set_xlim((0,1250))

        ax[1].plot(df[index].i,df[index].dayCab.rolling(30).mean(),label="Observed (30d)")
        ax[1].plot(df[index].i,df[index].Yp.rolling(30).mean(),label="Prediction avg.")
        ax[1].axvline(200)
        ax[1].axvline(800)
        #ax.plot(df[index].i,df[index].Yt,label="training")
        ax[1].legend()
        ax[1].set_xlim((0,1250))
        
        ax[0].set_title("Covariance prediction %(A)s - %(B)s"%locals() )
        #ax[1].set_title("Covariance average prediction %(A)s - %(B)s"%locals() )
        plt.savefig("%(A)s-%(B)s.png"%locals())
        plt.savefig("%(A)s-%(B)s.svg"%locals())
        plt.close()
    #    plt.show()
