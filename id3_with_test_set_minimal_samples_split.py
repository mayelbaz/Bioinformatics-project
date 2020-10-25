import sklearn
import io
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn import tree
import pickle


# from sklearn.model_selection import KFold, cross_val_score, cross_val_predict

def convert_to_int(word):
    new_value = ''.join(str(ord(c)) for c in word)
    return word.replace(word, new_value)


def read_vcf(path):
    with open(path, 'r') as f:
        lines = [l for l in f if not l.startswith('##')]
    return pd.read_table(
        io.StringIO(str.join(os.linesep, lines)),
        dtype={'#CHROM': str, 'POS': int, 'ID': str, 'RE,F': str, 'ALT': str,
               'QUAL': str, 'FILTER': str, 'INFO': str}
    ).rename(columns={'#CHROM': 'CHROM'})


if __name__ == '__main__':

    classification = ["TRUE", "TRUE", "TRUE", "TRUE", "TRUE", "FALSE", "TRUE", "FALSE", "TRUE", "TRUE", "TRUE", "TRUE",
                      "TRUE", "FALSE", "TRUE", "FALSE", "TRUE", "TRUE", "TRUE", "TRUE", "TRUE", "FALSE", "TRUE", "TRUE",
                      "TRUE", "TRUE", "TRUE", "TRUE", "TRUE", "TRUE", "TRUE", "TRUE", "FALSE", "TRUE", "TRUE", "FALSE",
                      "TRUE", "TRUE", "FALSE", "TRUE", "TRUE", "TRUE", "TRUE", "FALSE", "TRUE", "TRUE", "FALSE", "TRUE",
                      "FALSE", "TRUE", "TRUE", "TRUE", "TRUE", "TRUE", "FALSE", "TRUE", "TRUE", "TRUE", "TRUE", "TRUE",
                      "TRUE", "FALSE", "FALSE", "TRUE", "TRUE", "FALSE"]
    classification = np.array(classification)

    all_GTs_array = pickle.load(open("vcf_original_only_PASS_SNPs_QUAL15000_no_NULL.p", "rb"))
    all_GTs_train = all_GTs_array[0:49].copy()
    all_GTs_test = all_GTs_array[49:66].copy()
    # all_GTs_array =np.array(all_GTs)
    kf = KFold(n_splits=4)

    for i in range(2, 11):
        score_j = 0
        print("Results for id3_with_test_set_minimal_samples_split={}".format(i))
        for j in range(0, 10):
            clf_tree = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=i)
            score = 0
            total_conf_mat = np.zeros((2, 2))
            for train, test in kf.split(all_GTs_train):
                # test1=all_GTs_array[train, :]
                # test2= classification[train]
                clf_tree.fit(all_GTs_train[train, :], classification[train])
                score += clf_tree.score(all_GTs_train[test, :], classification[test])
                y_pred = clf_tree.predict(all_GTs_train[test, :])
                conf_mat = confusion_matrix(classification[test], y_pred)
            avg_score = score / 4
            score_j += avg_score

            # print(conf_mat)
            # total_conf_mat += conf_mat
        print(score_j / 10)
        # print(total_conf_mat)
        # print(all_GTs)
