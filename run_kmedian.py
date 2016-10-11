import glob # linux style path finder
import pandas as pd
import numpy as np
from kmedian import kmedian
import os 
import sys

# get the training data matrix for a certain (appliance, date)
# from all homes 
def get_train_dataframe(files, curr_date):
    train_df = pd.DataFrame()
    for f in files:
        df1 = pd.read_csv(f, index_col='Date')
        try:
            new_row = df1.loc[curr_date.isoformat()]
            # print 'new_row', new_row
            train_df = train_df.append(new_row, ignore_index=True)
        except KeyError:
            print 'no date', curr_date, 'in file', f
    return train_df


# the smallest "time shift absolute error" distance (TSAE)
def select_pattern(patterns, usage, delta):
    nrow, ncol = patterns.shape
    usage_m = np.tile(usage, (nrow, 1))
    dist_m = np.zeros(shape=(nrow, ncol/delta))
    for i in range(dist_m.shape[1]):
          np.fabs(np.roll(patterns, i*delta, axis=1) - usage_m).sum(axis=1, out=dist_m[:,i])
    # print "distance matrix", dist_m
    row, col = np.unravel_index(dist_m.argmin(), dist_m.shape)
    return np.roll(patterns[row], col*delta)


# run median on test and train data
def run_kmedian(testdir, traindir, soldir, k_clusters):
    """
    :param testdir: test data directory, run_kmedian will run through all .csv files under the directory
    :param traindir: train data directory, file name format: 'home_id'_'appliance'_'weekday/weekend'_out.csv
    :param soldir: solution output directory, file name format: 'home_id'_'weekend/weekday'_solution.csv
    :param k_clusters: number of clusters for kmedian, by default 47
    :return: void
    """

    appliances = ['air', 'dryer', 'furnace', 'lights_plugs', 'refrigerator']

    for doc in os.listdir(testdir):
        if doc.endswith('.csv'):
            home_id = doc.split('_')[0]
            testfile = os.path.join(testdir, doc)
            test = pd.read_csv(testfile)
            solfile = os.path.join(soldir, home_id+'_sol.csv')
            solution = pd.DataFrame()
            if os.path.exists(solfile):
                os.remove(solfile)

            test['localminute'] = pd.to_datetime(test.localminute)
            test['Date'] = test.localminute.dt.date
            test['Time'] = test.localminute.dt.time
            test.drop('localminute', axis=1, inplace=True)
            test = test.pivot(index='Date', columns='Time', values='use').fillna(0)

            for curr_date in pd.date_range('2013-07-01', periods=10, freq='D'):
                curr_date = curr_date.date()
                # test_day is numpy.array
                test_day = test.loc[curr_date].values
                # solution_day is pandas.DataFrame
                solution_day = pd.DataFrame({'timestamp': pd.date_range(curr_date, periods=1440, freq='min'),
                                             'use': test_day})

                if curr_date.weekday() < 5:
                    day_of_week = 'weekday'
                else:
                    day_of_week = 'weekend'
                for app in appliances:
                    # prepare training data matrix for current date
                    glob_path = os.path.join(traindir, '*_' + app + '_' + day_of_week + '_out.csv')
                    train_all_files = glob.glob(glob_path)
                    train = get_train_dataframe(train_all_files, curr_date)

                    # train:
                    # kmedian uses numpy.array
                    kpatterns = kmedian(np.array(train), k=k_clusters, steps=10)
                    # print "k patterns", kpatterns

                    # predict:
                    # predict on the test_day with TASE
                    # the prediction is actually to select one pattern from all k patterns
                    pattern = select_pattern(kpatterns, usage=test_day, delta=10)
                    # print "selected pattern", pattern

                    # concat the pattern
                    solution_day[app] = pattern
                    print 'solution for', curr_date, app, pattern

                    # subtract the predicted appliance from the total usage
                    test_day = np.subtract(test_day, pattern)
                    # print 'residual use', test_day

                # concat the prediction of current date
                solution = pd.concat([solution, solution_day], axis=0, ignore_index=True)
            print 'final solution', solution.head(20)
            solution.to_csv(solfile)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print 'need 4 arguments, given', len(sys.argv)
    run_kmedian(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), int(sys.argv[4]))






