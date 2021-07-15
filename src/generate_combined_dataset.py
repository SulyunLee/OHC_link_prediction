import pandas as pd

def generate_cet_dataset(com_df, emb_df, textsim_df):
        cet_combined_df = pd.concat([com_df, emb_df[['BC_emb','GD_emb','MB_emb','PM_emb']]], axis=1)
        cet_combined_df = pd.concat([cet_combined_df, textsim_df[['BC_Textsim','GD_Textsim','MB_Textsim']]], axis=1)

        cet_combined_df = cet_combined_df[['pair', 'BC_PA', 'BC_AA', 'BC_JC', 'BC_CM', 'BC_CLP', 'BC_emb', 'BC_Textsim',
                                        'GD_PA', 'GD_AA', 'GD_JC', 'GD_CM', 'GD_CLP', 'GD_emb', 'GD_Textsim',
                                        'MB_PA', 'MB_AA', 'MB_JC', 'MB_CM', 'MB_CLP', 'MB_emb', 'MB_Textsim',
                                        'PM_PA', 'PM_AA', 'PM_JC', 'PM_CM', 'PM_CLP', 'PM_emb', 'label']]
        
        return cet_combined_df
    
def generate_comemb_dataset(com_df, emb_df):
    comemb_combined_df = pd.concat([com_df, emb_df[['BC_emb', 'GD_emb', 'MB_emb', 'PM_emb']]], axis=1)
    comemb_combined_df = comemb_combined_df[['pair', 'BC_PA', 'BC_AA', 'BC_JC', 'BC_CM', 'BC_CLP', 'BC_emb', 'GD_PA', 'GD_AA', 'GD_JC', 'GD_CM', 'GD_CLP', 'GD_emb', 'MB_PA', 'MB_AA', 'MB_JC', 'MB_CM', 'MB_CLP', 'MB_emb','PM_PA', 'PM_AA', 'PM_JC', 'PM_CM', 'PM_CLP', 'PM_emb', 'label']]

    return comemb_combined_df

if __name__ == "__main__":
    com_folder = "data/proposed_com/"
    emb_folder = "data/proposed_emb/"
    textsim_folder = "data/proposed_textsim/"

    start_week = 50
    end_week = 101

    for i in range(end_week - start_week - 1):
    # for i in range(1):
        print('Generating week {} train and test dataset...'.format(i+start_week))
        com_train_filename = "{}bax_week{}_train.csv".format(com_folder, i+start_week)
        emb_train_filename = "{}bax_week{}_train.csv".format(emb_folder, i+start_week)
        # textsim_train_filename = "{}bax_week{}_train.csv".format(textsim_folder, i+start_week)

        com_test_filename = "{}bax_week{}_test.csv".format(com_folder, i+start_week)
        emb_test_filename = "{}bax_week{}_test.csv".format(emb_folder, i+start_week)
        # textsim_test_filename = "{}bax_week{}_test.csv".format(textsim_folder, i+start_week)
        
        com_train = pd.read_csv(com_train_filename)
        emb_train = pd.read_csv(emb_train_filename)
        # textsim_train = pd.read_csv(textsim_train_filename)

        com_test = pd.read_csv(com_test_filename)
        emb_test = pd.read_csv(emb_test_filename)
        # textsim_test = pd.read_csv(textsim_test_filename)
        
        print('Writing combined datasets to csv file...')
        # cet_combined_train = generate_cet_dataset(com_train, emb_train, textsim_train)
        # cet_combined_train.to_csv("data/proposed_cet/bax_week{}_train.csv".format(i+start_week), index=False)
        # cet_combined_test = generate_cet_dataset(com_test, emb_test, textsim_test)
        # cet_combined_test.to_csv("data/proposed_cet/bax_week{}_test.csv".format(i+start_week), index=False)

        # comemb_combined_train = generate_comemb_dataset(com_train, emb_train)
        # comemb_combined_train.to_csv("data/proposed_comemb/bax_week{}_train.csv".format(i+start_week), index=False)
        comemb_combined_test = generate_comemb_dataset(com_test, emb_test)
        comemb_combined_test.to_csv("data/proposed_comemb/bax_week{}_test.csv".format(i+start_week), index=False)





