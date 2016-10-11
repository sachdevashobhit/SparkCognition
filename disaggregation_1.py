
# coding: utf-8

# In[ ]:

import glob # use your path
import pandas as pd
from sklearn.cluster import KMeans
import scipy
import numpy as np
import datetime
import re

class Disaggregation:

    

    @staticmethod
    def preparedata(file1,zh):
        data2 = pd.DataFrame()
        for i in file1:
        print i
        data = pd.read_csv(i)
        # data.drop(['Date'], axis=1, inplace=True)
            data.drop(data.columns[[0, 1]], axis=1, inplace=True)
        data = data.fillna(0)
        data1 = pd.DataFrame(data.iloc[zh])
        data2 = pd.concat([data1,data2],axis=1)
        print data2, "data2"
        return data2
    
    @staticmethod    
    def kmeansdata(data2):
        da2 = data2.transpose()
        k_mea= KMeans(init='k-means++', n_clusters=30)
    #     k_mea= KMeans(k=47)
        barc = k_mea.fit(da2)
        fdf2 = pd.DataFrame(barc.cluster_centers_)
        fdf3 = fdf2.transpose()
        print fdf3, "fdf3 inside function"
        return fdf3
    
    @staticmethod
    def takeabs(fdf3,actual):
        delta_t = 12
        jh = 288/delta_t
        ans = []
        for restf in fdf3.columns:
        a = fdf3[restf]
        for shift in range(jh):
            ans.append(scipy.absolute(scipy.roll(a,shift*delta_t) - actual).sum())
        indexloc = ans.index(min(ans))
        pstar = indexloc/jh
        jstar= indexloc%jh
        adf=fdf3[pstar]
        adff=np.roll(adf,jstar)
        adff1=pd.Series(adff)
        print adff1,"pd.Series(adff)"
        return adff1
    
    @staticmethod
    def mergdat(mergpat,bestpat):
        bestpat = pd.concat([mergpat,bestpat], axis=1)
        return bestpat
    
    @staticmethod
    def diaggragate():
        d1 = datetime.datetime.now()
        files = glob.glob('test/*.csv')
        for j in files:
        home_id = re.findall(r'\d+', j.split('_')[0])[0]
        ite = 0
        words = ['air','dryer','furnace','lights_plugs','refrigerator']
        # for readit in range(0,14400,1440):
        for readit in range(0,2880,288):
        testf = pd.read_csv(j, skiprows=readit , nrows=288)
        bestpatmp = pd.DataFrame()
        actual = testf.iloc[:,7]
        print actual
        for  i in words:
            ghj = '*_'+i+'_weekday_out.csv'
            allFiles = glob.glob(ghj)
            data3 = preparedata(allFiles,ite)
            kmpat = kmeansdata(data3)
            bestpat = takeabs(kmpat,actual)
            actual = actual.subtract(bestpat)
            print actual, "actual"
            bestpatmp = mergdat(bestpatmp,bestpat)
        bestpatmp.to_csv('solution_'+home_id+'.csv', mode='a', index = False, index_label = False, header=False)
        ite += 1


        d2 = datetime.datetime.now()

        print d2 - d1

