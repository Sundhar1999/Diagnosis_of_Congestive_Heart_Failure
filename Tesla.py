import import_ipynb
import testing_wavelet as tw
import pandas as pd
import wfdb as w
import numpy as np
import math
#importing Jupyter notebook from testing_wavelet.ipynb


temp = []
trdf = []
for i in range(1,4):
    mitrec = w.rdrecord("mit/"+str(i),sampto = 10000)
    mitann = w.rdann('mit/'+str(i), 'atr', sampto=10000)
    df = pd.DataFrame(mitrec.p_signal)
    temp,prd  = tw.denoise(df)
    print(str(prd)+" is prd")
    snr = 40-(20 * (np.log10(prd)))
    print(str(snr)+" of "+str(i))
    trdf.append(temp)
    
    
rdf = pd.DataFrame(trdf)

crdfm = rdf.transpose()
crdfm

trdf = []
for i in range(1,4):
    mitrec = w.rdrecord("chf/chf0"+str(i),sampto = 10000)
    mitann = w.rdann('chf/chf0'+str(i), 'ecg', sampto=10000)
    df = pd.DataFrame(mitrec.p_signal)
    temp,prd  = tw.denoise(df)
    print(str(prd)+" is prd")
    snr = 40-(20 * (np.log10(prd)))
    print(str(snr)+" of "+str(i))
    trdf.append(temp)
    
rdf = pd.DataFrame(trdf)

crdfchf = rdf.transpose()
crdfchf

# trdf = []
# for i in range(1,3):
#     mitrec = w.rdrecord("chf/chf0"+str(i),sampto = 9000)
#     mitann = w.rdann('chf/chf0'+str(i), 'ecg', sampto=9000)
#     df = pd.DataFrame(mitrec.p_signal)
#     trdf,prd = tw.denoise(df)
    
#     print(str(prd)+" is prd")
    
# #     nm = np.sum(np.power(trdf,2))
# #     dm = np.sum(np.power(np.subtract(trdf,df[0]),2))
    
# #     snr = 10 * (np.log10(nm/dm))
#     snr = 40-(20 * (np.log10(prd)))
    
#     print(str(snr)+" of "+str(i))

