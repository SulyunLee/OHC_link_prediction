'''
Author: Sulyun Lee
This script generates the dataset based on three baseline features (PA, AA, JC) for supervised modeling of link prediction.
It constructs five baseline models and one proposed model
* Baseline models:
    - three baseline features based on aggregated network
    - three baseline features based on BC channel
    - three baseline features based on GD channel
    - three baseline features based on MB channel
    - three baseline features based on PM channel
* Proposed model:
    - 12 features generated based on each channel
'''
#----------------Import libraries-------------------
import pandas as pd
import numpy as np
from datetime import datetime
import networkx as nx
import collections
from tqdm import tqdm
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

def preferential_attachment(G, instances):
    '''
    This function computes the preferential attachment index for node pairs.
    Input:
      - G: the graph to compute the preferential attachment. 
           This graph should include the nodes that are in the instances.
      - instances: the list of tuples that include the node pairs to compute the preferntial attachment.
    Output:
      - pa: the numpy array of length instances that includes the preferential attachment values for each instance.
    '''
    pa = np.zeros((len(instances))).astype(int)
    pa_generator = nx.algorithms.link_prediction.preferential_attachment(G, instances)
    # append the PA values for each instance.
    for i, line in enumerate(pa_generator):
        pa[i] = line[2]

    return pa

def adamic_adar(G, instances):
    '''
    This function computes the Adamic/Adar index for node pairs.
    Input:
      - G: The networkx graph to compute the Adamic/Adar index
           This grah should include the nodes that are in the instances.
      - instance: the list of tuples that include the node pairs to compute the AA.
    Output:
      - aa: the numpy array of length instances that includes the AA values for each instance
    '''
    aa = np.zeros((len(instances)))
    aa_generator = nx.algorithms.link_prediction.adamic_adar_index(G, instances)
    # append the AA values for each instance.
    for i, line in enumerate(aa_generator):
        aa[i] = line[2]

    return aa

def jaccard(G, instances):
    '''
    This function computes the Jaccard coefficient for node pairs.
    Input:
      - G: The networkx graph to compute the Jaccard coefficient.
           This graph should include the nodes that are in the instances
      - instance: the list of tuples that include the node pairs to compute the Jaccard.
    Output:
      - jc: the numpy array of length instances that includes the Jaccard coefficient values for each instance
    '''
    jc = np.zeros((len(instances)))
    jc_generator = nx.algorithms.link_prediction.jaccard_coefficient(G, instances)
    # append the JC values for each instance
    for i, line in enumerate(jc_generator):
        jc[i] = line[2]

    return jc

def generate_2hop_instances(graph):
        two_hop_generator = nx.all_pairs_shortest_path(graph, 2)
        pairs = {}
        for node in two_hop_generator:
            author = node[0]
            two_hop_authors = list(node[1].keys())
            # Remove 0 hop (self pair)
            pairs[author] = two_hop_authors[1:]

        instances = []
        for node in pairs:
            for node2 in pairs[node]:
                if aggregated_graph.has_edge(node, node2):
                    continue
                else:
                    instances.append((node, node2))

        return instances

def construct_dataset_weekly(instances, feature_graph, label_graph):
    train_dict = {'pair':instances}
    test_dict = {'pair':instances}

    # Construct train dataset
    pa = preferential_attachment(feature_graph, instances)
    train_dict['PA'] = pa
    aa = adamic_adar(feature_graph, instances)
    train_dict['AA'] = aa
    jc = jaccard(feature_graph, instances)
    train_dict['JC'] = jc

    label = np.zeros((len(instances))).astype(bool)
    for idx, pair in enumerate(instances):
        if label_graph.has_edge(pair[0], pair[1]):
            label[idx] = True

    train_dict['label'] = label

    train_df = pd.DataFrame(train_dict)

    return train_df

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

    summary_dict = {'Aggregated_new_edges':[], 'BC_new_edges':[], 'GD_new_edges':[], 'MB_new_edges':[], 'PM_new_edges':[]}

    # for i in range(end_week - start_week):
    for i in range(1):

        # construct graphs based on each channel
        print('Making week{} train and test data...'.format(start_week+i))
        bc_graph = construct_graph(bc_weeks_subset[i])
        gd_graph = construct_graph(gd_weeks_subset[i])
        mb_graph = construct_graph(mb_weeks_subset[i])
        pm_graph = construct_graph(pm_weeks_subset[i])

        print('Generating first model...')
        # construct this week aggregated network - trainset feature
        aggregated_df = pd.concat([bc_weeks_subset[i], gd_weeks_subset[i], mb_weeks_subset[i], pm_weeks_subset[i]], ignore_index=True, sort=False) 
        aggregated_graph = construct_graph(aggregated_df)

        # construct next week aggregated network - trainset label, testset feature
        nextweek_agg_df = pd.concat([bc_weeks_subset[i+1], gd_weeks_subset[i+1], mb_weeks_subset[i+1], pm_weeks_subset[i+1]],ignore_index=True,  sort=False)
        nextweek_agg_graph = construct_graph(nextweek_agg_df)

        # construct next next week aggregated network - testset label
        nextnextweek_agg_df = pd.concat([bc_weeks_subset[i+2], gd_weeks_subset[i+2], mb_weeks_subset[i+2], pm_weeks_subset[i+2]],ignore_index=True, sort=False)
        nextnextweek_agg_graph = construct_graph(nextnextweek_agg_df)

        # make instances for train aggregated network
        train_instances = generate_2hop_instances(aggregated_graph)

        ## Baseline models with 3 features in aggregated network 
        train_df = construct_dataset_weekly(train_instances, aggregated_graph, nextweek_agg_graph)

        test_instances = generate_2hop_instances(nextweek_agg_graph)
        # consider only new links in test dataset
        test_new_instances = set(test_instances) - set(train_instances)
        test_new_instances = list(test_new_instances)

        test_df = construct_dataset_weekly(test_new_instances, nextweek_agg_graph, nextnextweek_agg_graph)
        summary_dict['Aggregated_new_edges'].append(test_df.label.sum())

        print('Saving dataset to csv file...')
        train_df.to_csv('data/baseline_agg/bax_week{}_train.csv'.format(start_week+i), index=False)

        test_df.to_csv('data/baseline_agg/bax_week{}_test.csv'.format(start_week+i), index=False)

        print('Generating second model...')
        # make instances for train BC network
        nextweek_bc_graph = construct_graph(bc_weeks_subset[i+1])
        nextnextweek_bc_graph = construct_graph(bc_weeks_subset[i+2])

        bc_train_instances = generate_2hop_instances(bc_graph)
        bc_train_df = construct_dataset_weekly(bc_train_instances, bc_graph, nextweek_bc_graph)
        
        bc_test_instances = generate_2hop_instances(nextweek_bc_graph)
        bc_test_new_instances = set(bc_test_instances) - set(bc_train_instances)
        bc_test_new_instances = list(bc_test_new_instances)
        bc_test_df = construct_dataset_weekly(bc_test_new_instances, nextweek_bc_graph, nextnextweek_bc_graph)

        summary_dict['BC_new_edges'].append(bc_test_df.label.sum())
        
        print('Saving dataset to csv file...')
        bc_train_df.to_csv('data/baseline_BC/bax_week{}_train.csv'.format(start_week+i), index=False)
        bc_test_df.to_csv('data/baseline_BC/bax_week{}_test.csv'.format(start_week+i), index=False)

        print('Generating third model...')
        # make instances for train GD network
        nextweek_gd_graph = construct_graph(gd_weeks_subset[i+1])
        nextnextweek_gd_graph = construct_graph(gd_weeks_subset[i+2])

        gd_train_instances = generate_2hop_instances(gd_graph)
        gd_train_df = construct_dataset_weekly(gd_train_instances, gd_graph, nextweek_gd_graph)
        
        gd_test_instances = generate_2hop_instances(nextweek_gd_graph)
        gd_test_new_instances = set(gd_test_instances) - set(gd_train_instances)
        gd_test_new_instances = list(gd_test_new_instances)
        gd_test_df = construct_dataset_weekly(gd_test_new_instances, nextweek_gd_graph, nextnextweek_gd_graph)

        summary_dict['GD_new_edges'].append(gd_test_df.label.sum())
        print('Saving dataset to csv file...')
        gd_train_df.to_csv('data/baseline_GD/bax_week{}_train.csv'.format(start_week+i), index=False)
        gd_test_df.to_csv('data/baseline_GD/bax_week{}_test.csv'.format(start_week+i), index=False)
        
        print('Generating fourth model...')
        # make instances for train MB network
        nextweek_mb_graph = construct_graph(mb_weeks_subset[i+1])
        nextnextweek_mb_graph = construct_graph(mb_weeks_subset[i+2])

        mb_train_instances = generate_2hop_instances(mb_graph)
        mb_train_df = construct_dataset_weekly(mb_train_instances, mb_graph, nextweek_mb_graph)
        
        mb_test_instances = generate_2hop_instances(nextweek_mb_graph)
        mb_test_new_instances = set(mb_test_instances) - set(mb_train_instances)
        mb_test_new_instances = list(mb_test_new_instances)
        mb_test_df = construct_dataset_weekly(mb_test_new_instances, nextweek_mb_graph, nextnextweek_mb_graph)

        summary_dict['MB_new_edges'].append(mb_test_df.label.sum())
        print('Saving dataset to csv file...')
        mb_train_df.to_csv('data/baseline_MB/bax_week{}_train.csv'.format(start_week+i), index=False)
        mb_test_df.to_csv('data/baseline_MB/bax_week{}_test.csv'.format(start_week+i), index=False)

        print('Generating fifth model...')
        # make instances for train PM network
        nextweek_pm_graph = construct_graph(pm_weeks_subset[i+1])
        nextnextweek_pm_graph = construct_graph(pm_weeks_subset[i+2])

        pm_train_instances = generate_2hop_instances(pm_graph)
        pm_train_df = construct_dataset_weekly(pm_train_instances, pm_graph, nextweek_pm_graph)
        
        pm_test_instances = generate_2hop_instances(nextweek_pm_graph)
        pm_test_new_instances = set(pm_test_instances) - set(pm_train_instances)
        pm_test_new_instances = list(pm_test_new_instances)
        pm_test_df = construct_dataset_weekly(pm_test_new_instances, nextweek_pm_graph, nextnextweek_pm_graph)

        summary_dict['PM_new_edges'].append(pm_test_df.label.sum())
        print('Saving dataset to csv file...')
        pm_train_df.to_csv('data/baseline_PM/bax_week{}_train.csv'.format(start_week+i), index=False)
        pm_test_df.to_csv('data/baseline_PM/bax_week{}_test.csv'.format(start_week+i), index=False)

    summary_df = pd.DataFrame(summary_dict)
    summary_df.to_csv('summary_statistics.csv')



