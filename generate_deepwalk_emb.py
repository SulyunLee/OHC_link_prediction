
#----------------Import libraries-------------------
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
#--------------------------------------------------

def compute_emb_similarity(instance, emb_dict):
    '''
    This function computesthe similarity based on embedding for the given pair of nodes
    Input:
      - instance: Tuple. The name of two nodes are in this tuple.
      - emb_dict: dictionary that includes the node names as the keys and the embedding
                  representations as the values.
    Output:
      - 0 or similarity: If either of nodes do not exist in the dictionary, then return 0
                         and otherwise, return float type of similarity value.
    '''
    if instance[0] in emb_dict and instance[1] in emb_dict:
        node1_rep = np.array(emb_dict[instance[0]]).reshape(1,-1)
        node2_rep = np.array(emb_dict[instance[1]]).reshape(1,-1)
        similarity = float(cosine_similarity(node1_rep, node2_rep))
        return similarity
    # if either of nodes do not exist in the graph, return 0
    else:
        return 0

def make_embedding_feature(row, emb_dict):
    '''
    This function generates the embbeding based features.
    This function is applied to each row of a dataframe using progress_apply or apply built-in functions
    Input:
      - row: the row of the dataframe (not specified when this function is called)
      - emb_dict: the dictionary that includes the nodes as keys and learned representations as values. 
    '''
    instance = eval(row['pair']) # the tuple of two nodes in 'pair' column of the dataframe
    sim = compute_emb_similarity(instance, emb_dict)

    return sim
def generate_channel_emb_sim(emb_file, input_df, col_name):

    emb_dict = emb_file.set_index(0).T.to_dict('list')
    input_df[col_name] = input_df.progress_apply(make_embedding_feature, args=[emb_dict], axis=1)

    return input_df
    




if __name__ == "__main__":
    # initialize parameters
    start_week = 50
    end_week = 101

    for i in range(end_week - start_week - 1):
    # for i in range(1):
        # read dataframe for pairs
        input_train_df = pd.read_csv('data/proposed_model/bax_week{}_train.csv'.format(start_week+i))
        input_test_df = pd.read_csv('data/proposed_model/bax_week{}_test.csv'.format(start_week+i))


        # read embedded representation files for each channel (trainset)
        bc_emb_train = pd.read_csv('../dataset/emb_channel_weekly/deepwalk_week{}_BC_char_train_128.emb'.format(i+start_week), header=None)
        gd_emb_train = pd.read_csv('../dataset/emb_channel_weekly/deepwalk_week{}_GD_char_train_128.emb'.format(i+start_week),header=None)
        mb_emb_train = pd.read_csv('../dataset/emb_channel_weekly/deepwalk_week{}_MB_char_train_128.emb'.format(i+start_week),header=None)
        pm_emb_train = pd.read_csv('../dataset/emb_channel_weekly/deepwalk_week{}_PM_char_train_128.emb'.format(i+start_week),header=None)

        # read embedded representation files for each channel (testset)
        bc_emb_test = pd.read_csv('../dataset/emb_channel_weekly/deepwalk_week{}_BC_char_train_128.emb'.format(i+start_week+1), header=None)
        gd_emb_test = pd.read_csv('../dataset/emb_channel_weekly/deepwalk_week{}_GD_char_train_128.emb'.format(i+start_week+1),header=None)
        mb_emb_test = pd.read_csv('../dataset/emb_channel_weekly/deepwalk_week{}_MB_char_train_128.emb'.format(i+start_week+1),header=None)
        pm_emb_test = pd.read_csv('../dataset/emb_channel_weekly/deepwalk_week{}_PM_char_train_128.emb'.format(i+start_week+1),header=None)

        print('Making week{} embedding features for train and test data...'.format(start_week+i))
        tqdm.pandas()

        emb_feature_train_df = generate_channel_emb_sim(bc_emb_train, input_train_df, 'BC_emb')
        emb_feature_train_df = generate_channel_emb_sim(gd_emb_train, emb_feature_train_df, 'GD_emb')
        emb_feature_train_df = generate_channel_emb_sim(mb_emb_train, emb_feature_train_df, 'MB_emb')
        emb_feature_train_df = generate_channel_emb_sim(pm_emb_train, emb_feature_train_df, 'PM_emb')
        
        emb_feature_test_df = generate_channel_emb_sim(bc_emb_test, input_test_df, 'BC_emb')
        emb_feature_test_df = generate_channel_emb_sim(gd_emb_test, emb_feature_test_df, 'GD_emb')
        emb_feature_test_df = generate_channel_emb_sim(mb_emb_test, emb_feature_test_df, 'MB_emb')
        emb_feature_test_df = generate_channel_emb_sim(pm_emb_test, emb_feature_test_df, 'PM_emb')

        emb_feature_train_df = emb_feature_train_df[['pair', 'BC_PA', 'BC_AA', 'BC_JC', 'GD_PA', 'GD_AA', 'GD_JC', 'MB_PA','MB_AA', 'MB_JC', 'PM_PA', 'PM_AA', 'PM_JC','BC_emb','GD_emb', 'MB_emb', 'PM_emb','label']]
        emb_feature_test_df = emb_feature_test_df[['pair', 'BC_PA', 'BC_AA', 'BC_JC', 'GD_PA', 'GD_AA', 'GD_JC', 'MB_PA','MB_AA', 'MB_JC', 'PM_PA', 'PM_AA', 'PM_JC','BC_emb','GD_emb', 'MB_emb', 'PM_emb','label']]
        print('Writing dataset to csv file...')
        emb_feature_train_df.to_csv('data/proposed_emb/bax_week{}_train.csv'.format(start_week+i), index=False)
        emb_feature_test_df.to_csv('data/proposed_emb/bax_week{}_test.csv'.format(start_week+i), index=False)
        

