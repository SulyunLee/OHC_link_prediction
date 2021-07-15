'''
Author: Sulyun Lee
This script generates the graph statistics for each channel.
'''

import networkx as nx
import pandas as pd
from generate_statistics import *

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

    # concatenate the list of dataframe into one dataframe
    bc_concat_df = pd.concat(bc_weeks_subset)
    gd_concat_df = pd.concat(gd_weeks_subset)
    mb_concat_df = pd.concat(mb_weeks_subset)
    pm_concat_df = pd.concat(pm_weeks_subset)
    agg_concat_df = pd.concat([bc_concat_df, gd_concat_df, mb_concat_df, pm_concat_df], ignore_index=True, sort=False)

    # Construct networkx graph from the dataframe
    bc_g = construct_graph(bc_concat_df)
    gd_g = construct_graph(gd_concat_df)
    mb_g = construct_graph(mb_concat_df)
    pm_g = construct_graph(pm_concat_df)
    agg_g = construct_graph(agg_concat_df)

    # Generate the statistics for each graph and export to csv file
    stat_dict = {}
    stat_dict['BC'] = generate_graph_statistics(bc_g)
    stat_dict['GD'] = generate_graph_statistics(gd_g)
    stat_dict['MB'] = generate_graph_statistics(mb_g)
    stat_dict['PM'] = generate_graph_statistics(pm_g)
    stat_dict['AGG'] = generate_graph_statistics(agg_g)

    stat_df = pd.DataFrame(stat_dict)
    stat_df.index = ['n', 'm', 'k_mean','k_max','std','cc','c','assortativity','n_giant','m_giant']
    stat_df.to_csv("summary/summary_statistics_channel.csv")



