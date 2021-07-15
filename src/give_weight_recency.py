'''
Created by Sulyun Lee
This script gives weights to each edge list based on the recency of
the activation of edges.
'''
# ------------IMPORT LIBRARIES----------------------
import pandas as pd
import numpy as np
from datetime import datetime
import networkx as nx
import math
#-----------------------------------------------------

#-------------INITIALIZE PARAMETERS--------------------
filename = '../dataset/BC_df_modified.csv'

#-------------------------------------------------------

network_df = pd.read_csv(filename, header=0)
network_df['Time'] = pd.to_datetime(network_df['Time'])
#network_df = network_df.rename(columns={'from':'source', 'to':'target'})


edge_dict = {} # for saving the last activated timestamps
               # keys: edge pair, values: list of last activated timestamps

weights = np.ones(network_df.shape[0]) # array for estimated weights for each edge pair
i=0

def compute(row):

    '''
    This function is applied to each row in dataframe.
    Give weights to each formed edge pair.
    Weight = sum(exponential of time difference between
    the last activated time of edge pair and current time).
    '''
    global i
    time = row.Time
    source = row.source
    target = row.target
    if ((source, target) in edge_dict) or ((target, source) in edge_dict): # if edge pair existed previously
        try:
            last_activated = edge_dict[(source, target)]
        except:
            last_activated = edge_dict[(target, source)]
        weight = sum([math.exp(-(time - last).days) for last in last_activated])
        weights[i] = weight
        try:
            edge_dict[(source, target)].append(time)
        except:
            edge_dict[(target, source)].append(time)
    else:
        edge_dict[(source, target)] = [time]
    if i % 10000==0:
        print(i)
    i+=1

# apply compute function to each row of network_df
network_df.apply(compute, axis=1)
network_df['weight'] = weights
# write the dataframe to csv
network_df.to_csv('../dataset/BC_weighted_modified.csv', index=False)

# reference: https://engineering.upside.com/a-beginners-guide-to-optimizing-pandas-code-for-speed-c09ef2c6a4d6
'''
network_arr = np.array(network_df)
time = network_arr[:,0]
source = network_arr[:,1]
target = network_arr[:,2]

cond1 = (network_df['source']==source) & (network_df['target']==target) & (network_df['Time'] < time)
cond2 = (network_df['target']==source) & (network_df['source']==target) & (network_df['Time'] < time)
previous_df = network_df.loc[cond1 | cond2]
if previous_df.shape[0] == 0:
    pass
else:
    weight = sum(np.exp(-(np.datetime64(time) - previous_df.Time).apply(lambda x:x.days)))
    weights[i] = weight
print(i)
i += 1
       
#network_df['weight'] = np.apply_along_axis(compute2, 1, network_arr)
'''
