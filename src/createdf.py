
import pandas as pd
from datetime import datetime


data = pd.read_csv('aggdata26.csv',nrows=5000)
df=pd.DataFrame({'Timestamp':data['localminute'],
                'Value':data['air'],
                'Houseid':'26'})


# In[106]:

for i in range(len(df['Timestamp'])):
    d=datetime.strptime(df['Timestamp'][i], "%m/%d/%Y %H:%M")
    df['Timestamp'][i]=d


# In[109]:

df['weekday'] = df['Timestamp'].dt.dayofweek


# In[111]:

df['Date']=df['Timestamp'].dt.date
df['Time']=df['Timestamp'].dt.time


# In[113]:

dt=df[df['weekday']<5].pivot(index='Date', columns='Time', values='Value')  
dt['Houseid']='26'


# In[116]:

dtw=df[df['weekday']>4].pivot(index='Date', columns='Time', values='Value')  
dtw['Houseid']='26'


# In[ ]:



