'''
Author: Sulyun Lee
This script generates the community-based and embedding-based features for the input dataset.
community-based features: community membership features based on label propagation and modularity maximization
embedding-based features: the similarity between the embedded vector representations.
'''

#----------------Import libraries-------------------
import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
#--------------------------------------------------

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

def compute_emb_similarity(instance, emb_df):
    if instance[0] in emb_df.values[:,0] and instance[1] in emb_df.values[:,0]:
        node1_rep = emb_df.loc[emb_df.values[:,0] == instance[0],1:]
        node2_rep = emb_df.loc[emb_df.values[:,0] == instance[1],1:]
        similarity = float(cosine_similarity(node1_rep, node2_rep))
        return similarity
    # if either of nodes do not exist in the graph, return 0
    else:
        return 0

def make_embedding_feature(row, emb_df):

    instance = eval(row['pair'])
    sim = compute_emb_similarity(instance, emb_df)

    return sim
    



if __name__ == "__main__":
    #--------------Initialize parameters------------------
    bc_file = '../dataset/BC_df_modified_withmsg.csv'
    gd_file = '../dataset/GD_df_modified_withmsg.csv'
    mb_file = '../dataset/MB_df_modified_withmsg.csv'
    pm_file = '../dataset/bilateral_PM_df.csv'

    # Input datafile with baseline features
    data_dir = 'data/proposed_model/'

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

    for i in range(end_week - start_week - 1):
    # for i in range(1):

        input_train_df = pd.read_csv(data_dir + 'bax_week{}_train.csv'.format(start_week+i))
        input_test_df = pd.read_csv(data_dir + 'bax_week{}_test.csv'.format(start_week+i))

        print('Making week{} train and test data...'.format(start_week+i))
        bc_graph = construct_graph(bc_weeks_subset[i])
        gd_graph = construct_graph(gd_weeks_subset[i])
        mb_graph = construct_graph(mb_weeks_subset[i])
        pm_graph = construct_graph(pm_weeks_subset[i])

        # BC
        nextweek_bc_graph = construct_graph(bc_weeks_subset[i+1])
        nextnextweek_bc_graph = construct_graph(bc_weeks_subset[i+2])
        # GD
        nextweek_gd_graph = construct_graph(gd_weeks_subset[i+1])
        nextnextweek_gd_graph = construct_graph(gd_weeks_subset[i+2])
        # MB
        nextweek_mb_graph = construct_graph(mb_weeks_subset[i+1])
        nextnextweek_mb_graph = construct_graph(mb_weeks_subset[i+2])
        # PM
        nextweek_pm_graph = construct_graph(pm_weeks_subset[i+1])
        nextnextweek_pm_graph = construct_graph(pm_weeks_subset[i+2])

        # Read embedded files for each channel (trainset)
        bc_emb_train = pd.read_csv('../dataset/emb_channel_weekly/week{}_unweighted_BC_train_32.emb'.format(i+start_week), sep=' ', skiprows=1, header=None)
        gd_emb_train = pd.read_csv('../dataset/emb_channel_weekly/week{}_unweighted_GD_train_32.emb'.format(i+start_week), sep=' ', skiprows=1, header=None)
        mb_emb_train = pd.read_csv('../dataset/emb_channel_weekly/week{}_unweighted_MB_train_32.emb'.format(i+start_week), sep=' ', skiprows=1, header=None)
        pm_emb_train = pd.read_csv('../dataset/emb_channel_weekly/week{}_unweighted_PM_train_32.emb'.format(i+start_week), sep=' ', skiprows=1, header=None)

        # Read embedded files for each channel (testset)
        bc_emb_test = pd.read_csv('../dataset/emb_channel_weekly/week{}_unweighted_BC_train_32.emb'.format(i+start_week+1), sep=' ', skiprows=1, header=None)
        gd_emb_test = pd.read_csv('../dataset/emb_channel_weekly/week{}_unweighted_GD_train_32.emb'.format(i+start_week+1), sep=' ', skiprows=1, header=None)
        mb_emb_test = pd.read_csv('../dataset/emb_channel_weekly/week{}_unweighted_MB_train_32.emb'.format(i+start_week+1), sep=' ', skiprows=1, header=None)
        pm_emb_test = pd.read_csv('../dataset/emb_channel_weekly/week{}_unweighted_PM_train_32.emb'.format(i+start_week+1), sep=' ', skiprows=1, header=None)


        tqdm.pandas()
        input_train_df['BC_emb'] = input_train_df.progress_apply(make_embedding_feature, args=[bc_emb_train], axis=1)
        input_train_df['GD_emb'] = input_train_df.progress_apply(make_embedding_feature, args=[gd_emb_train], axis=1)
        input_train_df['MB_emb'] = input_train_df.progress_apply(make_embedding_feature, args=[mb_emb_train], axis=1)
        input_train_df['PM_emb'] = input_train_df.progress_apply(make_embedding_feature, args=[pm_emb_train], axis=1)

        input_test_df['BC_emb'] = input_test_df.progress_apply(make_embedding_feature, args=[bc_emb_test], axis=1)
        input_test_df['GD_emb'] = input_test_df.progress_apply(make_embedding_feature, args=[gd_emb_test], axis=1)
        input_test_df['MB_emb'] = input_test_df.progress_apply(make_embedding_feature, args=[mb_emb_test], axis=1)
        input_test_df['PM_emb'] = input_test_df.progress_apply(make_embedding_feature, args=[pm_emb_test], axis=1)

        input_train_df = input_train_df[['BC_PA','BC_AA','BC_JC','BC_emb','GD_PA','GD_AA','GD_JC','GD_emb','MB_PA','MB_AA','MB_JC','MB_emb','PM_PA','PM_AA','PM_JC','PM_emb']]
        input_test_df = input_test_df[['BC_PA','BC_AA','BC_JC','BC_emb','GD_PA','GD_AA','GD_JC','GD_emb','MB_PA','MB_AA','MB_JC','MB_emb','PM_PA','PM_AA','PM_JC','PM_emb']]

        input_train_df.to_csv('data/proposed_emb/bax_week{}_train.csv'.format(start_week+i), index=False)
        input_test_df.to_csv('data/proposed_emb/bax_week{}_test.csv'.format(start_week+i), index=False)
