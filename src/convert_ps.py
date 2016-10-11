from __future__ import print_function, division
import numpy as np
import pandas as pd

from nilmtk.utils import get_datastore
from nilmtk.utils import check_directory_exists
from nilmtk.datastore import Key
from nilmtk.measurement import LEVEL_NAMES
from nilm_metadata import convert_yaml_to_hdf5, save_yaml_to_datastore

from os.path import join, isdir, isfile
from os import listdir
import re

MAJOR_LOAD = ["use","air","furnace","light_plugs","oven","fridge","car","others"]
TIME_INDEX = "localminute"


def convert_ps(ps_path, output_path, out_format="HDF"):
    # open datastore
    store = get_datastore(output_path, out_format, mode="w")

    # TODO: check 'US/Central'
    data_path = join(ps_path, "data")
    _convert_to_datastore(data_path, store, 'US/Central')

    # add metadata
    meta_path = join(ps_path, "meta")
    save_yaml_to_datastore(meta_path, store)

    store.close()

    print ("Done converting Pecan Street to HDF5")


def _convert_to_datastore(input_path, store, tz):
    check_directory_exists(input_path)
    homes = _get_all_homes(input_path)
    for home in homes:
        home_id = int(re.search("home_([\d]*).csv", home).group(1))
        csv_filename = join(input_path, home)
        dtype_dict = {m: np.float32 for m in MAJOR_LOAD}
        dtype_dict[TIME_INDEX] = pd.datetime
        whole_df = pd.read_csv(csv_filename, index_col=TIME_INDEX, dtype=dtype_dict)
        del whole_df.index.name
        print ("processing ", home_id, end="... ")
        for meter in MAJOR_LOAD:
            meter_id = int(MAJOR_LOAD.index(meter))+1
            table_key = Key(building=home_id, meter=meter_id)
            table_df = _load_csv(whole_df, meter, tz)
            table_df.sort_index()
            store.put(str(table_key), table_df)
            print (meter, end=" ")
        print ("finished", end="!")
        print ()


def _get_all_homes(input_path):
    files = [p for p in listdir(input_path) if isfile(join(input_path, p))]
    regex = 'home_(\d\d?).csv'
    p = re.compile(regex)
    homes = []
    for f in files:
        m = p.match(f)
        if m:
            homes.append(m.group())
    homes.sort()
    return homes


def _load_csv(whole_df, meter, tz):
    df = whole_df[[meter]]
    # print ("index name here:", df.index.name)
    two_level_index = pd.MultiIndex.from_tuples([('power','active')])
    df.columns = two_level_index
    df.columns.set_names(LEVEL_NAMES, inplace=True)
    df.to_csv("test.csv")
    # df = df.tz_convert(tz)

    return df
