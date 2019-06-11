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


def compute_emb_similarity(instance, emb_dict):
    if instance[0] in emb_dict and instance[1] in emb_dict:
        node1_rep = np.array(emb_dict[instance[0]]).reshape(1,-1)
        node2_rep = np.array(emb_dict[instance[1]]).reshape(1,-1)
        similarity = float(cosine_similarity(node1_rep, node2_rep))
        return similarity
    # if either of nodes do not exist in the graph, return 0
    else:
        return 0

def make_embedding_feature(row, emb_df):

    instance = eval(row['pair'])
    sim = compute_emb_similarity(instance, emb_df)

    return sim

'''
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
'''
    



if __name__ == "__main__":
    #--------------Initialize parameters------------------
    # Input datafile with baseline features
    data_dir = 'data/proposed_model/'

    start_week = 50
    end_week = 101
    #----------------------------------------------------


    # for i in range(end_week - start_week - 1):
    for i in range(1):

        input_train_df = pd.read_csv(data_dir + 'bax_week{}_train.csv'.format(start_week+i))
        input_test_df = pd.read_csv(data_dir + 'bax_week{}_test.csv'.format(start_week+i))

        print('Making week{} train and test data...'.format(start_week+i))

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
        bc_dict = bc_emb_train.set_index(0).T.to_dict('list')
        input_train_df['BC_emb'] = input_train_df.progress_apply(make_embedding_feature, args=[bc_dict], axis=1)
        gd_dict = gd_emb_train.set_index(0).T.to_dict('list')
        input_train_df['GD_emb'] = input_train_df.progress_apply(make_embedding_feature, args=[gd_dict], axis=1)
        mb_dict = mb_emb_train.set_index(0).T.to_dict('list')
        input_train_df['MB_emb'] = input_train_df.progress_apply(make_embedding_feature, args=[mb_dict], axis=1)
        pm_dict = pm_emb_train.set_index(0).T.to_dict('list')
        input_train_df['PM_emb'] = input_train_df.progress_apply(make_embedding_feature, args=[pm_dict], axis=1)

        bc_dict = bc_emb_test.set_index(0).T.to_dict('list')
        input_test_df['BC_emb'] = input_test_df.progress_apply(make_embedding_feature, args=[bc_dict], axis=1)
        gd_dict = gd_emb_test.set_index(0).T.to_dict('list')
        input_test_df['GD_emb'] = input_test_df.progress_apply(make_embedding_feature, args=[gd_dict], axis=1)
        mb_dict = mb_emb_test.set_index(0).T.to_dict('list')
        input_test_df['MB_emb'] = input_test_df.progress_apply(make_embedding_feature, args=[mb_dict], axis=1)
        pm_dict = pm_emb_test.set_index(0).T.to_dict('list')
        input_test_df['PM_emb'] = input_test_df.progress_apply(make_embedding_feature, args=[pm_dict], axis=1)

        input_train_df = input_train_df[['BC_PA','BC_AA','BC_JC','BC_emb','GD_PA','GD_AA','GD_JC','GD_emb','MB_PA','MB_AA','MB_JC','MB_emb','PM_PA','PM_AA','PM_JC','PM_emb']]
        input_test_df = input_test_df[['BC_PA','BC_AA','BC_JC','BC_emb','GD_PA','GD_AA','GD_JC','GD_emb','MB_PA','MB_AA','MB_JC','MB_emb','PM_PA','PM_AA','PM_JC','PM_emb']]

        input_train_df.to_csv('data/proposed_emb/bax_week{}_train.csv'.format(start_week+i), index=False)
        input_test_df.to_csv('data/proposed_emb/bax_week{}_test.csv'.format(start_week+i), index=False)