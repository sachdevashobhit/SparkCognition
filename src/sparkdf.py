import pandas as pd
from datetime import datetime
import os


def get_data_frames(path):
    output_path = os.path.join(path, "df_output")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    for doc in os.listdir(path):
        if doc.endswith(".csv"):
            data=pd.read_csv(os.path.join(path,doc))
            home_id = doc.split('_')[0]
            for column in data.columns[2:]:
                df=pd.DataFrame({'Timestamp':data['localminute'],
                            'Value':data[column],
                            'Houseid':home_id})
                df['Timestamp']=pd.to_datetime(df['Timestamp'])
                df['weekday'] = df['Timestamp'].dt.dayofweek
                df['Date']=df['Timestamp'].dt.date
                df['Time']=df['Timestamp'].dt.time

                dt=df[df['weekday']<5].pivot(index='Date', columns='Time', values='Value')  
                dt['Houseid']=home_id
                dtw=df[df['weekday']>4].pivot(index='Date', columns='Time', values='Value')  
                dtw['Houseid']=home_id


                dt.to_csv(os.path.join(output_path, home_id+"_"+column+"_weekday_out.csv"))

                dtw.to_csv(os.path.join(output_path, home_id+"_"+column+"_weekend_out.csv"))
