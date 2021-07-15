'''
Social Network Analytics Final Project
Author: Sulyun Lee, Hankyu Jang
Description:
    This script is for modeling the supervised approach based on the training dataset.
    We use the following supervised approach: random forest
    Then, evaluate the models based on the testing dataset.
'''
import argparse
import pandas as pd
import numpy as np
# import sklearn.metrics
from sklearn.metrics import precision_score, accuracy_score, recall_score, roc_curve, auc
from sklearn.ensemble import AdaBoostClassifier

from pathlib import Path
from tqdm import tqdm
import math

def evaluate(y_true, y_pred):
    accuracy =  accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    # fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_value = auc(fpr, tpr)
    return np.array([accuracy, precision, recall, auc_value])

def dcg_score(element):
    score = 0.0
    for order, rank in enumerate(element):
        score += float(rank) / math.log((order+2))

    return score

def evaluate_topk(y_true, y_pred_prob, model, k):
    probs = y_pred_prob[:,model.classes_ == True].flatten()
    sorted_idx = np.argsort(probs)[::-1]
    topk_label = y_true[sorted_idx[:k]]
    topk_pred = np.ones(k).astype(bool)

    # precision at k
    precision_k = precision_score(topk_label, topk_pred)

    # Normalized dicounted cumulative gain (nDCG) at k
    best = dcg_score(topk_label)
    dcg = dcg_score(topk_pred)
    ndcg = best / dcg

    return np.array([precision_k, ndcg])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Supervised learning for link prediction')
    parser.add_argument('-sweek', '--startweek', type=int, default=50,
                        help= 'training week starting week')
    parser.add_argument('-eweek', '--endweek', type=int, default=100,
                        help= 'training week ending week')
    parser.add_argument('-folder', '--folder', type=str, default="baseline_agg",
                        help= 'folder containing the files')
    args = parser.parse_args()

    startweek = args.startweek
    endweek = args.endweek
    folder = args.folder

    data_folder = Path("data/{}".format(folder))
    write_folder = Path("result/{}".format(folder))

    num_metrics = 4
    train_results = np.zeros((endweek-startweek, num_metrics))
    test_results = np.zeros((endweek-startweek, num_metrics))

    k_list = [10, 20 ,50]
    topk_results = np.zeros((endweek-startweek, len(k_list)*2))

    for i, week in enumerate(tqdm(range(startweek, endweek))):
        train_filename = data_folder / "bax_week{}_train.csv".format(week)
        test_filename = data_folder / "bax_week{}_test.csv".format(week)

        # print("Loading training and testing data file...")
        train_df = pd.read_csv(train_filename)
        test_df = pd.read_csv(test_filename)

        train_features_df = train_df.iloc[:,1:-1]
        test_features_df = test_df.iloc[:,1:-1]

        X_train = np.array(train_features_df)
        X_test = np.array(test_features_df)

        y_train = np.array(train_df['label'])
        y_test = np.array(test_df['label'])

        clf = AdaBoostClassifier(n_estimators=50)
        try:
            clf.fit(X_train, y_train)
                #predict
            y_pred = clf.predict(X_test)
            y_pred_prob = clf.predict_proba(X_test)
            test_results[i] = evaluate(y_test, y_pred)

            for idx, k in enumerate(k_list):
                topk_results[i, idx*2: idx*2+2] = evaluate_topk(y_test, y_pred_prob, clf, k=k)
        except Exception as e:
            print(e)
            test_results[i] = np.array([-1,-1,-1,-1])
            topk_results[i] = np.array([-1,-1,-1,-1,-1,-1])


    column_names = ["All"]
    df_precision_test = pd.DataFrame(
            data = test_results[:,1],
            index = ["{}->{}".format(week+1, week+2) for week in range(startweek, endweek)],
            columns = column_names
            )
    df_recall_test = pd.DataFrame(
            data = test_results[:,2],
            index = ["{}->{}".format(week+1, week+2) for week in range(startweek, endweek)],
            columns = column_names
            )

    df_precision_test.to_csv(write_folder / "precision_ab_week_{}_{}.csv".format(startweek, endweek))
    df_recall_test.to_csv(write_folder / "recall_ab_week_{}_{}.csv".format(startweek, endweek))

    for idx, k in enumerate(k_list):
        # precision_k
        df = pd.DataFrame(
                data = topk_results[:,idx*2],
                index = ["{}->{}".format(week+1, week+2) for week in range(startweek, endweek)],
                columns = column_names
                )
        df.to_csv(write_folder / "top{}_precision_ab_week_{}_{}.csv".format(k, startweek, endweek))
        # ndcg
        df = pd.DataFrame(
                data = topk_results[:,idx*2+1],
                index = ["{}->{}".format(week+1, week+2) for week in range(startweek, endweek)],
                columns = column_names
                )
        df.to_csv(write_folder / "top{}_ndcg_ab_week_{}_{}.csv".format(k, startweek, endweek))

