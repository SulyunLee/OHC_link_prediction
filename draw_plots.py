# import argparse
import pandas as pd
import numpy as np
# import matplotlib as mpl
# mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv("summary/summary_week_50_100.csv")
    # Rename the first column
    df.rename(columns={'Unnamed: 0':'metric'}, inplace=True)
    
    # Add classifier to the dataframe
    classifier_list = ["rf", 
                       "logit",
                       "nb",
                       "ab",
                       "nn"
                       ]
    df["classifier"] = [item for item in classifier_list for _ in range(8)]
    # Remove results from naive bayes
    df = df[df["classifier"]!="nb"]

    df_max = df.groupby(["metric"]).max()

    df_max = df_max.reset_index()
    df_max.drop("classifier", axis=1, inplace=True)
    # Drop recall, top50prec, top50ndcg
    df_max.drop(df_max.index[[1,6,7]], inplace=True)

    df_max.plot.bar(x="metric", rot=0)
    plt.show()

