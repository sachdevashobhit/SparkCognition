  
# coding: utf-8

# In[6]:

import pandas as pd
import numpy as np
import glob
# reading files with actual consumption for each home ID. Class Error has one method do_error(path) 
class Error:
    
    # do_error takes path to test directory as a single parameter
    @staticmethod
    def do_error(path='test/*.csv'):
        files = glob.glob(path)
        print files
        k=' '
        for j in files:
            print len(j)    
            if j.endswith('__weekday_out.csv'):
            k = j[5:-17]
                frame1=pd.read_csv(j)
                # reading files with predicted data for each home ID
                frame2=pd.read_csv('solution_'+str(k)+'.csv')
                # Appliances for which consumption is predicted
                frame2.columns=['Air predicted','Dryer predicted','Furnace predicted','Light plugs predicted','Refrigerator predicted']
                frame2['Timestamp']=frame1['Timestamp']
                #Merging actual and predicted consumption into a data frame
                df2=pd.merge(frame1,frame2)
        #        print(df2)
        #        df=df2.drop(df2.columns[[0,6,7,8]],axis=1,inplace=True)
            # Dropping  header
                df=df2.drop(['Unnamed: 0','Timestamp','Total','weekday'], axis=1)
        #        print(df)
            # Initializing list for each error type eg. root mean square error, mean absolute percentage error, etc.
                errormp=[]
                mse=[]
                rmse=[]
                norm_error=[]
                dev=[]
            mape=[]
            # Appliances for which prediction error is calculated
                appliance=['Air','Dryer', 'Furnace','Light plugs','Refrigerator']
               # for i in range(len(df.columns)/2):
                for words in appliance:
                print(words)
                i=words
                    nzdf=[df[i],df[str(i)+" predicted"]]
                nzdf=pd.DataFrame(nzdf, columns='str(i)','str(i)+ predicted')
                print (type(nzdf))
                dev2=pd.DataFrame(np.array(df[i])-np.array(df[str(i)+" predicted"]))
                e2=(np.sum(np.array(dev2)**2))/len(dev2)
                rms=e2**0.5
                ne=(np.sum(np.absolute(np.array(dev2))))/np.sum(np.array(df[i]))
                      devabs=pd.DataFrame(np.array(nzdf[i])-np.array(nzdf[str(i)+" predicted"]))
                    ma=(np.sum(np.absolute(np.array(devabs))/np.array(nzdf[i])))*100
                mse.append(e2)
                rmse.append(rms)        
                norm_error.append(ne)
                    mape.append(ma)
                errormp=[mse,rmse,norm_error,mape]
                errormp=pd.DataFrame(errormp) 
                errormp.to_csv('error_'+str(k)+'.csv', mode='a', index = False, index_label = False, header=False)
