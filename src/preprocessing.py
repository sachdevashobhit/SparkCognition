import pandas as pd
from pandas import *
import pandas as pd
import os
import sys
import operator
import numpy as np
from collections import Counter
from datetime import datetime
import json
from datetime import datetime

class Preprocess:
    
    # This function creates a DataFrame for other functions appending columns with similar appliencies, e.g. air1 and air2
    # It takes the path to the cvs file as an argument
    @staticmethod
    def load_group(path):
        df = pd.read_csv(path)
        df.fillna(0, inplace=True)
        df['air'] = df['air1'] + df['air2'] +df['air3'] + df['airwindowunit1']
        df.drop(['air1','air2','air3', 'airwindowunit1'],inplace=True,axis=1,errors='ignore')

        df['aquarium'] = df['aquarium1']
        df.drop(['aquarium1'],inplace=True,axis=1,errors='ignore')
        df['bathroom'] = df['bathroom1'] + df['bathroom2']
        df.drop(['bathroom1','bathroom2'],inplace=True,axis=1,errors='ignore')
        df['bedroom'] = df['bedroom1'] + df['bedroom2'] + df['bedroom3'] + df['bedroom4'] + df['bedroom5']
        df.drop(['bedroom1','bedroom2', 'bedroom3', 'bedroom4', 'bedroom5'],inplace=True,axis=1,errors='ignore')
        df['car'] = df['car1']
        df.drop(['car1'],inplace=True,axis=1,errors='ignore')
        df['clotheswasher'] = df['clotheswasher1'] + df['clotheswasher_dryg1']
        df.drop(['clotheswasher1', 'clotheswasher_dryg1'],inplace=True,axis=1,errors='ignore')

        df['diningroom'] = df['diningroom1'] + df['diningroom2']
        df.drop(['diningroom1','diningroom2'],inplace=True,axis=1,errors='ignore')
        df['dishwasher'] = df['dishwasher1']
        df.drop(['dishwasher1'],inplace=True,axis=1,errors='ignore')
        df['disposal'] = df['disposal1']
        df.drop(['disposal1'],inplace=True,axis=1,errors='ignore')
        df['dryer'] = df['drye1'] + df['dryg1']
        df.drop(['drye1', 'dryg1'],inplace=True,axis=1,errors='ignore')

        df['freezer'] = df['freezer1']
        df.drop(['freezer1'],inplace=True,axis=1,errors='ignore')
        df['furnace'] = df['furnace1'] + df['furnace2']
        df.drop(['furnace1','furnace2'],inplace=True,axis=1,errors='ignore')
        df['garage'] = df['garage1'] + df['garage2']
        df.drop(['garage1','garage2'],inplace=True,axis=1,errors='ignore')
        df['heater'] = df['heater1']
        df.drop(['heater1'],inplace=True,axis=1,errors='ignore')
        df['housefan'] = df['housefan1']
        df.drop(['housefan1'],inplace=True,axis=1,errors='ignore')
        df['icemaker'] = df['icemaker1']
        df.drop(['icemaker1'],inplace=True,axis=1,errors='ignore')
        df['jacuzzi'] = df['jacuzzi1']
        df.drop(['jacuzzi1'],inplace=True,axis=1,errors='ignore')
        df['kitchen'] = df['kitchen1'] + df['kitchen2']
        df.drop(['kitchen1','kitchen2'],inplace=True,axis=1,errors='ignore')
        df['kitchenapp'] = df['kitchenapp1'] + df['kitchenapp2']
        df.drop(['kitchenapp1','kitchenapp2'],inplace=True,axis=1,errors='ignore')
        df['lights_plugs'] = df['lights_plugs1'] + df['lights_plugs2'] + df['lights_plugs3'] + df['lights_plugs4'] + df['lights_plugs5'] + df['lights_plugs6']
        df.drop(['lights_plugs1','lights_plugs2', 'lights_plugs3', 'lights_plugs4', 'lights_plugs5', 'lights_plugs6'],inplace=True,axis=1,errors='ignore')
        df['livingroom'] = df['livingroom1'] + df['livingroom2']
        df.drop(['livingroom1','livingroom2'],inplace=True,axis=1,errors='ignore')
        df['microwave'] = df['microwave1']
        df.drop(['microwave1'],inplace=True,axis=1,errors='ignore')
        df['office'] = df['office1']
        df.drop(['office1'],inplace=True,axis=1,errors='ignore')
        df['outsidelights_plugs'] = df['outsidelights_plugs1'] + df['outsidelights_plugs2']
        df.drop(['outsidelights_plugs1','outsidelights_plugs2'],inplace=True,axis=1,errors='ignore')
        df['oven'] = df['oven1'] + df['oven2']
        df.drop(['oven1','oven2'],inplace=True,axis=1,errors='ignore')
        df['pool'] = df['pool1'] + df['pool2'] + df['poollight1'] + df['poolpump1']
        df.drop(['pool1','pool2', 'poollight1', 'poolpump1'],inplace=True,axis=1,errors='ignore')
        df['pump'] = df['pump1']
        df.drop(['pump1'],inplace=True,axis=1,errors='ignore')
        df['range'] = df['range1']
        df.drop(['range1'],inplace=True,axis=1,errors='ignore')
        df['refrigerator'] = df['refrigerator1'] + df['refrigerator2']
        df.drop(['refrigerator1','refrigerator2'],inplace=True,axis=1,errors='ignore')
        df['security'] = df['security1']
        df.drop(['security1'],inplace=True,axis=1,errors='ignore')
        df['shed'] = df['shed1']
        df.drop(['shed1'],inplace=True,axis=1,errors='ignore')
        df['sprinkler'] = df['sprinkler1']
        df.drop(['sprinkler1'],inplace=True,axis=1,errors='ignore')
        df['unknown'] = df['unknown1'] + df['unknown2'] + df['unknown3'] + df['unknown4']
        df.drop(['unknown1','unknown2', 'unknown3', 'unknown4'],inplace=True,axis=1,errors='ignore')
        df['utilityroom'] = df['utilityroom1']
        df.drop(['utilityroom1'],inplace=True,axis=1,errors='ignore')
        df['venthood'] = df['venthood1']
        df.drop(['venthood1'],inplace=True,axis=1,errors='ignore')
        df['waterheater'] = df['waterheater1'] + df['waterheater2']
        df.drop(['waterheater1','waterheater2'],inplace=True,axis=1,errors='ignore')
        df['winecooler'] = df['winecooler1']
        df.drop(['winecooler1'],inplace=True,axis=1,errors='ignore')

        df.drop(df.columns[0:3], axis=1, inplace=True)

        return df


    

    # it will return a dataframe with only selected months
    # months are array of month strings: ['06','07','08']
    @staticmethod
    def get_months(df, months):
        dt = df[df['localminute'].str[5:7].isin(months)]
        dt.drop('localminute', axis=1, inplace=True)
        return dt    


    

    # get top k appliances that consume most energy
    @staticmethod
    def get_top_appliance(dt, k):
        top_list = dt.sum().order(ascending=False)
        return top_list[1:k+1].index  # top 1 is 'use'


    

    # scan every file, and top k appliance from all
    @staticmethod
    def get_top(path, k):
        top_list = list()
        for doc in os.listdir(path):
            if doc.endswith(".csv"):
                dt = get_months(load_group(os.path.join(path,doc)), 
                                ['06', '07','08'])
                top_list.extend(get_top_appliance(dt,k))
                print "processed file: ", doc
        return [app for app, count in Counter(top_list).most_common(k)]


    
class Transpose:
    # This function takes the directory with aggregated files as an input and outputs transposed csv files to use with 
    # K-means and K-medians into df_output1 directory. parameter 'number_d' is a number of minutes we take to do downsampling
    # if d is 1, then there is no downsampling

    @staticmethod
    def get_data_frames(path):
        
        output_path = os.path.join(path, "df_output")
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        for doc in os.listdir(path):                
            if doc.endswith(".csv"):               # loop through every csv file at the target directory
                data=pd.read_csv(os.path.join(path,doc) )
                home_id = doc.split('_')[0]        # get home_id from doc name (the part before '_', 0=number, 1=out.csv)
                print "processing ", doc
                for column in data.ix[:,2:]:       # subset from 3rd column to last column
                    df=pd.DataFrame({'Timestamp':data['localminute'],
                                'Value':data[column],
                                'Houseid':home_id})
                    # transform 'localminute' to meaningful datetime format
                    # e.g. transform 00:10:00-05 to 05:10:00
                    df['Timestamp']=pd.to_datetime(df['Timestamp'])
                    df.set_index(['Timestamp'], verify_integrity = True)

                    # create 3 new columns shows the corresponding dayofweek (0-6), date, and time
                    df['weekday'] = df['Timestamp'].dt.dayofweek
                    df['Date']=df['Timestamp'].dt.date
                    df['Time']=df['Timestamp'].dt.time 
        
                    # transform the matrix to the format we want
                    # weekday<5 represent weekdays
                    dt=df[df['weekday']<5].pivot(index='Date', columns='Time', values='Value').fillna(0)
                    #weekday>4 represents weekends
                    dtw=df[df['weekday']>4].pivot(index='Date', columns='Time', values='Value').fillna(0)
 
                    dt.to_csv(os.path.join(output_path, home_id + '_' + column + '_weekday_out.csv'))
                    dtw.to_csv(os.path.join(output_path, home_id + '_' + column + '_weekend_out.csv'))
                        

                        
                                                
                        
    # We give the output of transposing function to downsample each csv
    @staticmethod
    # Function to downsample (default - downsample to 5 minutes)
    def downsample(path, d=5):
        # output directory path
        output_path = os.path.join(path, "df_output_downsampled")
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        # get all csv from path and read using pandas
        for doc in os.listdir(path):
            if doc.endswith(".csv"):
                data=pd.read_csv(os.path.join(path,doc))
           
                #Creating data frame, dt with column 'Date'        
                dt=pd.DataFrame(data=data,columns=['Date'])
                # Taking the sum of d original observation which is the consumption at every minute
                # The data will be downsampled to d minutes (default 5)
                for i in range(0,288):
                    dt[data.columns[i*d+d]]=data[data.columns[i*d+1:i*d+(d+1)]].sum(1)
                
                # Creating csv with downsampled data to d minutes
                dt.to_csv(os.path.join(output_path,'downsampled_'+ doc))
                    
