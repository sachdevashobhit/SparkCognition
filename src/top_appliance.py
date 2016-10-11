
# coding: utf-8

# In[7]:

import pandas as pd
import os
import sys
import operator
import numpy as np
from collections import Counter
from datetime import datetime
import json


# In[8]:

# This function creates a DataFrame for other functions appending columns with similar appliencies, e.g. air1 and air2
# It takes the path to the cvs file as an argument
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


# In[9]:

# it will return a dataframe with only selected months
# months are array of month strings: ['06','07','08']
def get_months(df, months):
    dt = df[df['localminute'].str[5:7].isin(months)]
    dt.drop('localminute', axis=1, inplace=True)
    return dt    


# In[10]:

# get top k appliances that consume most energy
def get_top_appliance(dt, k):
    top_list = dt.sum().sort_values(ascending=False)
    return top_list[1:k+1].index  # top 1 is 'use'


# In[11]:

# scan every file, and top k appliance from all
def get_top(path, k):
    top_list = list()
    for doc in os.listdir(path):
        if doc.endswith(".csv"):
            dt = get_months(load_group(os.path.join(path,doc)), 
                            ['06', '07','08'])
            top_list.extend(get_top_appliance(dt,k))
            print "processed file: ", doc
    return [app for app, count in Counter(top_list).most_common(k)]


# In[12]:

def main(args):
    path = str(args[0])
    k = int(args[1])
    out = str(args[2])
    top_list = get_top(path, k)
    with open(out, 'w') as outfile:
        json.dump(top_list, outfile)


if __name__ == "__main__":
    main(sys.argv[1:])

