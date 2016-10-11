
import pandas as pd
from pandas import *
from datetime import datetime
import os
import pprint

#Using this class we split all the data into k brackets (k = 4). Class Bracketing has one static get_folders method
class Bracketing:
    #takes path to the directory with preprocessed files as an input. copies files into one of four brackets depending on the threshold values    
    @staticmethod
    def get_folders (path):
        
        output_path = os.path.join(path, "output")
        output_path1 = os.path.join(output_path, "bracket1")
        output_path2 = os.path.join(output_path, "bracket2")
        output_path3 = os.path.join(output_path, "bracket3")
        output_path4 = os.path.join(output_path, "bracket4")
        
        if not os.path.exists(output_path):
        os.mkdir(output_path)
        if not os.path.exists(output_path1):
        os.mkdir(output_path1)
        if not os.path.exists(output_path2):
        os.mkdir(output_path2)
        if not os.path.exists(output_path3):
        os.mkdir(output_path3)
        if not os.path.exists(output_path4):
        os.mkdir(output_path4)
        
        my_dict = dict()
        # for each file we define a maximum total usage and assign it to x variable
        for doc in os.listdir(path):
        if doc.endswith(".csv"):
            df = pd.read_csv(os.path.join(path,doc))
            df1 = df[df['localminute'].str[5:7].isin(['06', '07', '08'])]
            df1['localminute']=pd.to_datetime(df1['localminute'])
            df1['weekday'] = df1['localminute'].dt.dayofweek
            df2=df1[df1['weekday']<5]
            x = df2['use'].max()
            my_dict[doc] = x
           #depending on the value of x we move a file into a particular directory based on the threshold value
            if x >= 0 and x < 10:
                df1.to_csv(os.path.join(output_path1, doc.split('.')[0]+".csv"))
            elif x >= 10 and x < 15:
                df1.to_csv(os.path.join(output_path2, doc.split('.')[0]+".csv"))
            elif x >= 15 and x < 20:
                df1.to_csv(os.path.join(output_path3, doc.split('.')[0]+".csv"))
            elif x >= 20:
                df1.to_csv(os.path.join(output_path4, doc.split('.')[0]+".csv"))
        print(my_dict)         

get_folders('/home/cg37999/data')




