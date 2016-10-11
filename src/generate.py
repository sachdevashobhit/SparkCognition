
# coding: utf-8

# In[9]:

import os
import top_appliance as tp
import json


# In[12]:

def main(path, list_file):
    path = str(args[0])
    list_file = str(args[1])
    with open(list_file) as data_file:    
        top_list = json.load(data_file)
    print "received top list: ", top_list
    output_path = os.path.join(path, "output")
    if not os.path.exists(output_path): os.mkdir(output_path)
    for doc in os.listdir(path):
        if doc.endswith(".csv"):
            df = tp.load_group(os.path.join(path,doc))
            df1 = df[['localminute'] + top_list]
            df1.to_csv(os.path.join(output_path, doc.split('.')[0]+"_out.csv"))
            print "processed ", doc


# In[13]:

main("../data_working/data/", "../data_working/data/top_list.json")


# In[ ]:

if __name__ == "__main__":
    main(args[1:])

