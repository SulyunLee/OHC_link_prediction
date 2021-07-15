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

from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.label_propagation import label_propagation_communities, asyn_lpa_communities
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

def community_label(row, community_obj, graph):
    pair = eval(row['pair'])
    if pair[0] in graph.nodes() and pair[1] in graph.nodes():
        community_idx = next(i for i,v in enumerate(community_obj) if pair[0] in v)
        if pair[1] in community_obj[community_idx]:
            return 1
        else:
            return 0
    else:
        return 0

def label_prop(graph):
    label_prop_community_gen = label_propagation_communities(graph)
    label_prop_community_train = []
    for community in label_prop_community_gen:
        if len(community) > 1:
            label_prop_community_train.append(community)

    return label_prop_community_train

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
        # GD
        nextweek_gd_graph = construct_graph(gd_weeks_subset[i+1])
        # MB
        nextweek_mb_graph = construct_graph(mb_weeks_subset[i+1])
        # PM
        nextweek_pm_graph = construct_graph(pm_weeks_subset[i+1])

        tqdm.pandas()

        modularity_community_train = greedy_modularity_communities(bc_graph)
        input_train_df['BC_CM'] = input_train_df.progress_apply(community_label, community_obj=modularity_community_train, graph=bc_graph, axis=1)
        modularity_community_train = greedy_modularity_communities(gd_graph)
        input_train_df['GD_CM'] = input_train_df.progress_apply(community_label, community_obj=modularity_community_train, graph=gd_graph, axis=1)
        modularity_community_train = greedy_modularity_communities(mb_graph)
        input_train_df['MB_CM'] = input_train_df.progress_apply(community_label, community_obj=modularity_community_train, graph=mb_graph, axis=1)
        modularity_community_train = greedy_modularity_communities(pm_graph)
        input_train_df['PM_CM'] = input_train_df.progress_apply(community_label, community_obj=modularity_community_train, graph=pm_graph, axis=1)

        modularity_community_test = greedy_modularity_communities(nextweek_bc_graph)
        input_test_df['BC_CM'] = input_test_df.progress_apply(community_label, community_obj=modularity_community_test, graph=nextweek_bc_graph, axis=1)
        modularity_community_test = greedy_modularity_communities(nextweek_gd_graph)
        input_test_df['GD_CM'] = input_test_df.progress_apply(community_label, community_obj=modularity_community_test, graph=nextweek_gd_graph, axis=1)
        modularity_community_test = greedy_modularity_communities(nextweek_mb_graph)
        input_test_df['MB_CM'] = input_test_df.progress_apply(community_label, community_obj=modularity_community_test, graph=nextweek_mb_graph, axis=1)
        modularity_community_test = greedy_modularity_communities(nextweek_pm_graph)
        input_test_df['PM_CM'] = input_test_df.progress_apply(community_label, community_obj=modularity_community_test, graph=nextweek_pm_graph, axis=1)

        # Label propagation
        label_prop_community_train = label_prop(bc_graph)
        input_train_df['BC_CLP'] = input_train_df.progress_apply(community_label, community_obj=label_prop_community_train, graph=bc_graph, axis=1)
        label_prop_community_train = label_prop(gd_graph)
        input_train_df['GD_CLP'] = input_train_df.progress_apply(community_label, community_obj=label_prop_community_train, graph=gd_graph, axis=1)
        label_prop_community_train = label_prop(mb_graph)
        input_train_df['MB_CLP'] = input_train_df.progress_apply(community_label, community_obj=label_prop_community_train, graph=mb_graph, axis=1)
        label_prop_community_train = label_prop(pm_graph)
        input_train_df['PM_CLP'] = input_train_df.progress_apply(community_label, community_obj=label_prop_community_train, graph=pm_graph, axis=1)
        
        label_prop_community_test = label_prop(nextweek_bc_graph)
        input_test_df['BC_CLP'] = input_test_df.progress_apply(community_label, community_obj=label_prop_community_test, graph=nextweek_bc_graph, axis=1)
        label_prop_community_test = label_prop(nextweek_gd_graph)
        input_test_df['GD_CLP'] = input_test_df.progress_apply(community_label, community_obj=label_prop_community_test, graph=nextweek_gd_graph, axis=1)
        label_prop_community_test = label_prop(nextweek_mb_graph)
        input_test_df['MB_CLP'] = input_test_df.progress_apply(community_label, community_obj=label_prop_community_test, graph=nextweek_mb_graph, axis=1)
        label_prop_community_test = label_prop(pm_graph)
        input_test_df['PM_CLP'] = input_test_df.progress_apply(community_label, community_obj=label_prop_community_test, graph=pm_graph, axis=1)

        input_train_df = input_train_df[['pair', 'BC_PA', 'BC_AA', 'BC_JC', 'BC_CM', 'BC_CLP',
                                                     'GD_PA', 'GD_AA', 'GD_JC', 'GD_CM', 'GD_CLP',
                                                     'MB_PA', 'MB_AA', 'MB_JC', 'MB_CM', 'MB_CLP',
                                                     'PM_PA', 'PM_AA', 'PM_JC', 'PM_CM', 'PM_CLP', 'label']]
        input_test_df = input_test_df[['pair', 'BC_PA', 'BC_AA', 'BC_JC', 'BC_CM', 'BC_CLP',
                                                     'GD_PA', 'GD_AA', 'GD_JC', 'GD_CM', 'GD_CLP',
                                                     'MB_PA', 'MB_AA', 'MB_JC', 'MB_CM', 'MB_CLP',
                                                     'PM_PA', 'PM_AA', 'PM_JC', 'PM_CM', 'PM_CLP', 'label']]
        input_train_df.to_csv('data/proposed_com/bax_week{}_train.csv'.format(start_week+i), index=False)
        input_test_df.to_csv('data/proposed_com/bax_week{}_test.csv'.format(start_week+i), index=False)
