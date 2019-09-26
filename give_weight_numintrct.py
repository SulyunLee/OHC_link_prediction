'''
Author: Sulyun Lee
This script computes the weights for each pair of nodes by the number of interactions and computes the weighted version of baseline features (PA, AA, JC) for the supervised link prediction.
'''

#----------------Import libraries-------------------
import pandas as pd
import numpy as np
from datetime import datetime
import networkx as nx
from tqdm import tqdm
import math
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

def generate_weight_dict(graph, edgelist_df):
    weight_dict = {}
    for edge in graph.edges:
        node1 = edge[0]
        node2 = edge[1]
        weight_dict[(node1,node2)] = \
                {'weight':edgelist_df.loc[\
                    ((edgelist_df.source == node1)&(edgelist_df.target == node2)) | \
                    ((edgelist_df.source == node2)&(edgelist_df.target == node1)),].shape[0]}

    return weight_dict

def compute_weighted_AA(row, graph):
    '''
    This function is for computing the Adamic/Adar index with the weighted graph version.
    Ref: Murata, T., & Moriyasu, S. (2007). Link prediction of social networks based on weighted proximity measures. 
         Proceedings of the IEEE/WIC/ACM International Conference on Web Intelligence, WI 2007, 85–88. 
    '''
    instance = eval(row['pair'])
    node1 = instance[0]
    node2 = instance[1]

    aa_val = 0

    if (node1 in graph) and (node2 in graph):
        common_neighbors = nx.common_neighbors(graph, node1, node2)

        for z in common_neighbors:
            weighted_cn = (graph.edges[z,node1]['weight'] + graph.edges[z,node2]['weight']) / 2 
            
            popularity = 0
            for n in graph.edges(z):
                popularity += graph.edges[n[0],n[1]]['weight']

            aa_val += (weighted_cn * (1/math.log(popularity)))

    return aa_val

def compute_weighted_JC(row, graph):
    '''
    This function is for computing the Jaccard Coefficient index with the weighted graph version.
    '''
    instance = eval(row['pair'])
    node1 = instance[0]
    node2 = instance[1]
    jc_val = 0
    
    if (node1 in graph) and (node2 in graph):
        common_neighbors = nx.common_neighbors(graph, node1, node2)

        weighted_cn = 0
        for z in common_neighbors:
            weighted_cn = (graph.edges[z,node1]['weight'] + graph.edges[z,node2]['weight']) / 2 

        weighted_degree1 = 0
        weighted_degree2 = 0

        for a in graph.edges(node1):
            weighted_degree1 += graph.edges[a[0],a[1]]['weight']

        for b in graph.edges(node2):
            weighted_degree2 += graph.edges[b[0],b[1]]['weight']
            
        jc_val = weighted_cn / (weighted_degree1 + weighted_degree2)

    return jc_val

    
    

def compute_weighted_PA(row, graph):
    '''
    This function is for computing the Preferential Attachment index with the weighted graph version.
    '''
    instance = eval(row['pair'])
    node1 = instance[0]
    node2 = instance[1]
    pa_val = 0

    if (node1 in graph) and (node2 in graph):

        weighted_degree1 = 0
        weighted_degree2 = 0

        for a in graph.edges(node1):
            weighted_degree1 += graph.edges[a[0],a[1]]['weight']

        for b in graph.edges(node2):
            weighted_degree2 += graph.edges[b[0],b[1]]['weight']
            
        pa_val = weighted_degree1 * weighted_degree2

    return pa_val

def make_weighted_features(df, i):
    bc_graph = construct_graph(bc_weeks_subset[i])
    gd_graph = construct_graph(gd_weeks_subset[i])
    mb_graph = construct_graph(mb_weeks_subset[i])
    pm_graph = construct_graph(pm_weeks_subset[i])

    # generate dictionary for weight in each channel
    # key - node pair, value - the number of interactions
    bc_weight_dict = generate_weight_dict(bc_graph, bc_weeks_subset[i])
    gd_weight_dict = generate_weight_dict(gd_graph, gd_weeks_subset[i])
    mb_weight_dict = generate_weight_dict(mb_graph, mb_weeks_subset[i])
    pm_weight_dict = generate_weight_dict(pm_graph, pm_weeks_subset[i])

    # set the graph attribute with weights
    nx.set_edge_attributes(bc_graph, bc_weight_dict)
    nx.set_edge_attributes(gd_graph, gd_weight_dict)
    nx.set_edge_attributes(mb_graph, mb_weight_dict)
    nx.set_edge_attributes(pm_graph, pm_weight_dict)

    df_new = df[['pair']]
    tqdm.pandas()
    df_new['BC_weighted_AA'] = df.progress_apply(compute_weighted_AA, args=[bc_graph], axis=1)
    df_new['BC_weighted_PA'] = df.progress_apply(compute_weighted_PA, args=[bc_graph], axis=1)
    df_new['BC_weighted_JC'] = df.progress_apply(compute_weighted_JC, args=[bc_graph], axis=1)

    df_new['GD_weighted_AA'] = df.progress_apply(compute_weighted_AA, args=[gd_graph], axis=1)
    df_new['GD_weighted_PA'] = df.progress_apply(compute_weighted_PA, args=[gd_graph], axis=1)
    df_new['GD_weighted_JC'] = df.progress_apply(compute_weighted_JC, args=[gd_graph], axis=1)

    df_new['MB_weighted_AA'] = df.progress_apply(compute_weighted_AA, args=[mb_graph], axis=1)
    df_new['MB_weighted_PA'] = df.progress_apply(compute_weighted_PA, args=[mb_graph], axis=1)
    df_new['MB_weighted_JC'] = df.progress_apply(compute_weighted_JC, args=[mb_graph], axis=1)

    df_new['PM_weighted_AA'] = df.progress_apply(compute_weighted_AA, args=[pm_graph], axis=1)
    df_new['PM_weighted_PA'] = df.progress_apply(compute_weighted_PA, args=[pm_graph], axis=1)
    df_new['PM_weighted_JC'] = df.progress_apply(compute_weighted_JC, args=[pm_graph], axis=1)

    df_new['label'] = df['label']

    return df_new


if __name__ == "__main__":
    bc_file = '../dataset/BC_df_modified_withmsg.csv'
    gd_file = '../dataset/GD_df_modified_withmsg.csv'
    mb_file = '../dataset/MB_df_modified_withmsg.csv'
    pm_file = '../dataset/bilateral_PM_df.csv'
    proposed_model_folder = "data/proposed_model/"

    start_week = 50
    end_week = 101

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
        print('Computing weights for week {} train and test dataset...'.format(i+start_week))
        train_filename = "{}bax_week{}_train.csv".format(proposed_model_folder, i+start_week)
        test_filename = "{}bax_week{}_test.csv".format(proposed_model_folder, i+start_week)

        train_df = pd.read_csv(train_filename)
        test_df = pd.read_csv(test_filename)

        # construct graphs based on each channel
        print('Making week{} train and test data...'.format(start_week+i))
        train_df_new = make_weighted_features(train_df, i)
        train_df_new.to_csv("data/proposed_model_weighted/bax_week{}_train.csv".format(i+start_week), index=False)

        test_df_new = make_weighted_features(test_df, i+1)
        test_df_new.to_csv("data/proposed_model_weighted/bax_week{}_test.csv".format(i+start_week), index=False)


        




