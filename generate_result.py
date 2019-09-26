import argparse
import pandas as pd
import numpy as np

from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate summary table from the results')
    parser.add_argument('-sweek', '--startweek', type=int, default=50,
                        help= 'training week starting week')
    parser.add_argument('-eweek', '--endweek', type=int, default=100,
                        help= 'training week ending week')
    args = parser.parse_args()

    startweek = args.startweek
    endweek = args.endweek

    ######################################################################
    # Generate result1
    ######################################################################
    strategy_list = ["All"]

    folder_list = ["baseline_agg",
                   "baseline_BC",
                   "baseline_GD",
                   "baseline_MB",
                   "baseline_PM",
                   "proposed_model",
                   ]

    classifier_list = ["rf", 
                       "logit",
                       "ab",
                       "nn"
                       ]
    metric_list = ["precision",
                   "top10_precision",
                   "top10_ndcg",
                   "top20_precision",
                   "top20_ndcg",
                   "top50_precision",
                   "top50_ndcg",
                   "recall"]
    
    n_strategy = len(strategy_list)
    n_folder = len(folder_list)
    n_classifier = len(classifier_list)
    n_metric = len(metric_list)

    summary = np.zeros((n_classifier * n_metric, n_folder * n_strategy))

    for i, metric in enumerate(metric_list):
        for j, cf in enumerate(classifier_list):
            for k, folder in enumerate(folder_list):
                row_idx = i * n_classifier + j
                col_idx = k * n_strategy

                filename = "result/{}/{}_{}_week_{}_{}.csv".format(folder, metric, cf, startweek, endweek)
                df = pd.read_csv(filename)
                # If there is -1 in the dataframe, disregard that entry
                summary[row_idx, col_idx:col_idx+n_strategy] = list(df[df!=-1].mean())

    # column_names = strategy_list * n_folder
    column_names = folder_list
    # row_names = metric_list * n_classifier
    row_names = classifier_list * n_metric

    df_summary = pd.DataFrame(
            data = summary,
            index = row_names,
            columns = column_names
            )

    df_summary.to_csv("summary/result1.csv".format(startweek, endweek))

    ######################################################################
    # Generate result1
    ######################################################################
    strategy_list = ["All"]

    folder_list = ["proposed_model",
                   "proposed_com",
                   "proposed_emb",
                   "proposed_textsim",
                   "proposed_comemb",
                   "proposed_cet"
                   ]

    n_strategy = len(strategy_list)
    n_folder = len(folder_list)
    n_classifier = len(classifier_list)
    n_metric = len(metric_list)

    summary = np.zeros((n_classifier * n_metric, n_folder * n_strategy))

    for i, metric in enumerate(metric_list):
        for j, cf in enumerate(classifier_list):
            for k, folder in enumerate(folder_list):
                row_idx = i * n_classifier + j
                col_idx = k * n_strategy

                filename = "result/{}/{}_{}_week_{}_{}.csv".format(folder, metric, cf, startweek, endweek)
                df = pd.read_csv(filename)
                # If there is -1 in the dataframe, disregard that entry
                summary[row_idx, col_idx:col_idx+n_strategy] = list(df[df!=-1].mean())

    # column_names = strategy_list * n_folder
    column_names = folder_list
    # row_names = metric_list * n_classifier
    row_names = classifier_list * n_metric

    df_summary = pd.DataFrame(
            data = summary,
            index = row_names,
            columns = column_names
            )

    df_summary.to_csv("summary/result2.csv".format(startweek, endweek))
