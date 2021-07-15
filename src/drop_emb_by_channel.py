'''
This script drops the embedding similarity column for each channel to compare the effects of the embedding for the channels.
'''


import argparse
import pandas as pd
import numpy as np
import modeling_rf
from tqdm import tqdm
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-sweek', '--startweek', type=int, default=50,
                        help= 'training week starting week')
    parser.add_argument('-eweek', '--endweek', type=int, default=100,
                        help= 'training week ending week')
    parser.add_argument('-channel', '--channel', type=str, default="BC",
                        help="The name of the channel to drop feature")

    args = parser.parse_args()

    startweek = args.startweek
    endweek = args.endweek
    channel = args.channel

    data_folder = Path("data/proposed_emb")
    write_folder = Path("data/proposed_emb_no{}".format(channel))

    for i, week in enumerate(tqdm(range(startweek, endweek))):
    # for i, week in enumerate(tqdm(range(50, 51))):

        train_filename = data_folder / "bax_week{}_train.csv".format(week)
        test_filename = data_folder / "bax_week{}_test.csv".format(week)

        train_df = pd.read_csv(train_filename)
        test_df = pd.read_csv(test_filename)

        drop_column_name = "{}_emb".format(channel)
        train_emb_dropped = train_df.drop(columns=[drop_column_name])
        test_emb_dropped = test_df.drop(columns=[drop_column_name])

        train_emb_dropped.to_csv(write_folder / "bax_week{}_train.csv".format(week), index=False)
        test_emb_dropped.to_csv(write_folder / "bax_week{}_test.csv".format(week), index=False)


    

