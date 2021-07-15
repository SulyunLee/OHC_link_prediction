
#----------------Import libraries-------------------
import pandas as pd
import numpy as np
from datetime import datetime
import networkx as nx
import collections
from tqdm import tqdm
import pickle
#----------------------------------------------------

def preprocess_networks(edgelist, start_week, end_week):
    '''
    This function preprocesses the network by dividing into weeks and selecting only useful weeks.
    Input:
      - edgelist: the dataframe that includes pairs of nodes and other columns
      - start_week: the integer which indicates the number of week to start with
      - end_week: the integer which indicates the number of week to end with
    Output:
      - weeks_subset: the list of dataframes. Each dataframe is the subset of edgelist divided into weekly basis.
    '''
    # remove duplicated rows
    edgelist = edgelist[~edgelist.duplicated()] # problem: later weeks do not include messages - should be included.
    # divide the edgelist into weeks
    edgelist['Time'] = pd.to_datetime(edgelist['Time'])
    weeks = [g for n, g in edgelist.groupby(pd.Grouper(key='Time', freq='W'))]

    # select only active weeks
    weeks_subset = weeks[start_week:end_week+1]

    return weeks_subset

def construct_graph(edgelist_df):
    '''
    This function constructs graph from the dataframe.
    The graph is undirected and unweighted.
    Input:
      - edgelist_df: the dataframe which includes pairs of nodes and other columns
    Output:
      - g: the networkx graph object that is constructed using edgelist_df.
    '''
    edgelist = edgelist_df[['source', 'target']] # extract only the source and target nodes
    g = nx.from_pandas_edgelist(edgelist) # construct the networkx graph using the dataframe
    
    return g
if __name__ == "__main__":

    #--------------Initialize parameters------------------
    bc_file = '../dataset/BC_df_modified_withmsg.csv'
    gd_file = '../dataset/GD_df_modified_withmsg.csv'
    mb_file = '../dataset/MB_df_modified_withmsg.csv'
    pm_file = '../dataset/bilateral_PM_df.csv'

    start_week = 50
    end_week = 101
    #----------------------------------------------------

    print('Reading data files...')
    # read the csv file for four channels
    bc_edgelist = pd.read_csv(bc_file)
    gd_edgelist = pd.read_csv(gd_file)
    mb_edgelist = pd.read_csv(mb_file)
    pm_edgelist = pd.read_csv(pm_file)

    #divide each channel into weeks
    bc_weeks_subset = preprocess_networks(bc_edgelist, start_week, end_week)
    gd_weeks_subset = preprocess_networks(gd_edgelist, start_week, end_week)
    mb_weeks_subset = preprocess_networks(mb_edgelist, start_week, end_week)
    pm_weeks_subset = preprocess_networks(pm_edgelist, start_week, end_week)

    bc_all = pd.concat(bc_weeks_subset, axis=0)
    gd_all = pd.concat(gd_weeks_subset, axis=0)
    mb_all = pd.concat(mb_weeks_subset, axis=0)
    pm_all = pd.concat(pm_weeks_subset, axis=0)

    entire_channel = pd.concat([bc_all[['source','target']], gd_all[['source','target']],mb_all[['source','target']],pm_all[['source','target']]], axis=0)
    entire_g = construct_graph(entire_channel)

    node_to_num_mapping = {}
    num_to_node_mapping = {}
    counter = 1
    for node in entire_g.nodes():
        node_to_num_mapping[node] = counter
        num_to_node_mapping[counter] = node
        counter += 1

    pickle.dump(node_to_num_mapping, open("node_to_num_mapping.pickle", "wb"))
    pickle.dump(num_to_node_mapping, open("num_to_node_mapping.pickle", "wb"))
