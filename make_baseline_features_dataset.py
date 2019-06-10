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

    flattened = [item for sublist in instances for item in sublist]
    node_set = set(flattened)
    nodes_not_included = [n for n in node_set if n not in feature_graph.nodes()]
    feature_graph.add_nodes_from(nodes_not_included)

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

    # for i in range(end_week - start_week-1):
    for i in range(12, 13):

        # construct graphs based on each channel
        print('Making week{} train and test data...'.format(start_week+i))
        bc_graph = construct_graph(bc_weeks_subset[i])
        gd_graph = construct_graph(gd_weeks_subset[i])
        mb_graph = construct_graph(mb_weeks_subset[i])
        pm_graph = construct_graph(pm_weeks_subset[i])

        print('Preparing graphs and instances for models...')
        # construct this week aggregated network - trainset feature
        aggregated_df = pd.concat([bc_weeks_subset[i], gd_weeks_subset[i], mb_weeks_subset[i], pm_weeks_subset[i]], ignore_index=True, sort=False) 
        aggregated_graph = construct_graph(aggregated_df)

        # construct next week aggregated network - trainset label, testset feature
        nextweek_agg_df = pd.concat([bc_weeks_subset[i+1], gd_weeks_subset[i+1], mb_weeks_subset[i+1], pm_weeks_subset[i+1]],ignore_index=True, sort=False)
        nextweek_agg_graph = construct_graph(nextweek_agg_df)

        # construct next next week aggregated network - testset label
        nextnextweek_agg_df = pd.concat([bc_weeks_subset[i+2], gd_weeks_subset[i+2], mb_weeks_subset[i+2], pm_weeks_subset[i+2]],ignore_index=True, sort=False)
        nextnextweek_agg_graph = construct_graph(nextnextweek_agg_df)

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

        # make instances for train aggregated network
        train_instances = generate_2hop_instances(aggregated_graph)
        # make instances for test aggregated network
        # consider only new links in test dataset
        test_instances = generate_2hop_instances(nextweek_agg_graph)
        test_new_instances = set(test_instances) - set(train_instances)
        test_new_instances = list(test_new_instances)

    ''' 
        print('Generating first model...')
        ## Baseline models with 3 features in aggregated network 
        train_df = construct_dataset_weekly(train_instances, aggregated_graph, nextweek_agg_graph)
        test_df = construct_dataset_weekly(test_new_instances, nextweek_agg_graph, nextnextweek_agg_graph)
        train_df.columns = ['pair','Agg_PA','Agg_AA','Agg_JC','label']
        test_df.columns = ['pair','Agg_PA','Agg_AA','Agg_JC','label']
        summary_dict['Aggregated_new_edges'].append(test_df.label.sum())

        print('Saving dataset to csv file...')
        train_df.to_csv('data/baseline_agg/bax_week{}_train.csv'.format(start_week+i), index=False)
        test_df.to_csv('data/baseline_agg/bax_week{}_test.csv'.format(start_week+i), index=False)

        print('Generating second model...')

        bc_train_df = construct_dataset_weekly(train_instances, bc_graph, nextweek_bc_graph)
        bc_test_df = construct_dataset_weekly(test_new_instances, nextweek_bc_graph, nextnextweek_bc_graph)
        bc_train_df.columns = ['pair','BC_PA','BC_AA','BC_JC','label'] 
        bc_test_df.columns = ['pair','BC_PA','BC_AA','BC_JC','label'] 

        summary_dict['BC_new_edges'].append(bc_test_df.label.sum())
        
        print('Saving dataset to csv file...')
        bc_train_df.to_csv('data/baseline_BC/bax_week{}_train.csv'.format(start_week+i), index=False)
        bc_test_df.to_csv('data/baseline_BC/bax_week{}_test.csv'.format(start_week+i), index=False)

        print('Generating third model...')

        gd_train_df = construct_dataset_weekly(train_instances, gd_graph, nextweek_gd_graph)
        
        gd_test_df = construct_dataset_weekly(test_new_instances, nextweek_gd_graph, nextnextweek_gd_graph)
        gd_train_df.columns = ['pair','GD_PA','GD_AA','GD_JC','label'] 
        gd_test_df.columns = ['pair','GD_PA','GD_AA','GD_JC','label'] 

        summary_dict['GD_new_edges'].append(gd_test_df.label.sum())

        print('Saving dataset to csv file...')
        gd_train_df.to_csv('data/baseline_GD/bax_week{}_train.csv'.format(start_week+i), index=False)
        gd_test_df.to_csv('data/baseline_GD/bax_week{}_test.csv'.format(start_week+i), index=False)
        
        print('Generating fourth model...')

        mb_train_df = construct_dataset_weekly(train_instances, mb_graph, nextweek_mb_graph)
        
        mb_test_df = construct_dataset_weekly(test_new_instances, nextweek_mb_graph, nextnextweek_mb_graph)
        mb_train_df.columns = ['pair','MB_PA','MB_AA','MB_JC','label'] 
        mb_test_df.columns = ['pair','MB_PA','MB_AA','MB_JC','label'] 

        summary_dict['MB_new_edges'].append(mb_test_df.label.sum())

        print('Saving dataset to csv file...')
        mb_train_df.to_csv('data/baseline_MB/bax_week{}_train.csv'.format(start_week+i), index=False)
        mb_test_df.to_csv('data/baseline_MB/bax_week{}_test.csv'.format(start_week+i), index=False)

        print('Generating fifth model...')

        pm_train_df = construct_dataset_weekly(train_instances, pm_graph, nextweek_pm_graph)
        
        pm_test_df = construct_dataset_weekly(test_new_instances, nextweek_pm_graph, nextnextweek_pm_graph)
        pm_train_df.columns = ['pair','PM_PA','PM_AA','PM_JC','label'] 
        pm_test_df.columns = ['pair','PM_PA','PM_AA','PM_JC','label'] 

        summary_dict['PM_new_edges'].append(pm_test_df.label.sum())

        print('Saving dataset to csv file...')
        pm_train_df.to_csv('data/baseline_PM/bax_week{}_train.csv'.format(start_week+i), index=False)
        pm_test_df.to_csv('data/baseline_PM/bax_week{}_test.csv'.format(start_week+i), index=False)

        print('Generating sixth model...')
        # Generate the proposed model with 12 features
        proposed_train_df = pd.DataFrame({'pair':train_instances})
        proposed_train_df = pd.concat([proposed_train_df, bc_train_df[['BC_PA','BC_AA','BC_JC']]], axis=1)
        proposed_train_df = pd.concat([proposed_train_df, gd_train_df[['GD_PA','GD_AA','GD_JC']]], axis=1)
        proposed_train_df = pd.concat([proposed_train_df, mb_train_df[['MB_PA','MB_AA','MB_JC']]], axis=1)
        proposed_train_df = pd.concat([proposed_train_df, pm_train_df[['PM_PA','PM_AA','PM_JC']]], axis=1)
        proposed_train_df = pd.concat([proposed_train_df, train_df[['label']]], axis=1)

        proposed_test_df = pd.DataFrame({'pair':test_new_instances})
        proposed_test_df = pd.concat([proposed_test_df, bc_test_df[['BC_PA','BC_AA','BC_JC']]], axis=1)
        proposed_test_df = pd.concat([proposed_test_df, gd_test_df[['GD_PA','GD_AA','GD_JC']]], axis=1)
        proposed_test_df = pd.concat([proposed_test_df, mb_test_df[['MB_PA','MB_AA','MB_JC']]], axis=1)
        proposed_test_df = pd.concat([proposed_test_df, pm_test_df[['PM_PA','PM_AA','PM_JC']]], axis=1)
        proposed_test_df = pd.concat([proposed_test_df, test_df[['label']]], axis=1)

        print('Saving dataset to csv file...')
        proposed_train_df.to_csv('data/proposed_model/bax_week{}_train.csv'.format(start_week+i), index=False)
        proposed_test_df.to_csv('data/proposed_model/bax_week{}_test.csv'.format(start_week+i), index=False)

    summary_df = pd.DataFrame(summary_dict)
    summary_df.to_csv('summary_statistics.csv')
    '''
    



