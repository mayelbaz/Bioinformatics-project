import io
import os
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
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
    classification=np.array(classification)

    # new_series = pd.DataFrame(classification)

    # data = read_vcf("test_sample.vcf")
    # data = read_vcf("CAGI_exome_hg19.gatk_reworked.vcf")
    data = pickle.load(open("vcf_original_only_PASS_SNPs.vcf", "rb"))
    headers = ["CAGI2012_CD_01", "CAGI2012_CD_02", "CAGI2012_CD_03", "CAGI2012_CD_04", "CAGI2012_CD_05",
               "CAGI2012_CD_06", "CAGI2012_CD_07", "CAGI2012_CD_08", "CAGI2012_CD_09", "CAGI2012_CD_10",
               "CAGI2012_CD_11", "CAGI2012_CD_12", "CAGI2012_CD_13", "CAGI2012_CD_14", "CAGI2012_CD_15",
               "CAGI2012_CD_16", "CAGI2012_CD_17", "CAGI2012_CD_18", "CAGI2012_CD_19", "CAGI2012_CD_20",
               "CAGI2012_CD_21", "CAGI2012_CD_22", "CAGI2012_CD_23", "CAGI2012_CD_24", "CAGI2012_CD_25",
               "CAGI2012_CD_26", "CAGI2012_CD_27", "CAGI2012_CD_28", "CAGI2012_CD_29", "CAGI2012_CD_30",
               "CAGI2012_CD_31", "CAGI2012_CD_32", "CAGI2012_CD_33", "CAGI2012_CD_34", "CAGI2012_CD_35",
               "CAGI2012_CD_36", "CAGI2012_CD_37", "CAGI2012_CD_38", "CAGI2012_CD_39", "CAGI2012_CD_40",
               "CAGI2012_CD_41", "CAGI2012_CD_42", "CAGI2012_CD_43", "CAGI2012_CD_44", "CAGI2012_CD_45",
               "CAGI2012_CD_46", "CAGI2012_CD_47", "CAGI2012_CD_48", "CAGI2012_CD_49", "CAGI2012_CD_50",
               "CAGI2012_CD_51", "CAGI2012_CD_52", "CAGI2012_CD_53", "CAGI2012_CD_54", "CAGI2012_CD_55",
               "CAGI2012CD_56", "CAGI2012_CD_57", "CAGI2012_CD_58", "CAGI2012_CD_59", "CAGI2012_CD_60",
               "CAGI2012_CD_61", "CAGI_2012_CD_62", "CAGI2012_CD_63", "CAGI2012_CD_64", "CAGI2012_CD_65",
               "CAGI2012_CD_66"]
    all_GTs = []
    for header in headers:
        extracted = data[header].str.split(':').str[0]
        new_df = extracted.to_frame(name=header)
        extracted = list(extracted)
        # for entry in extracted:
        #     entry = int(convert_to_int(entry))
        all_GTs.append(extracted)
    all_GTs = pd.DataFrame(all_GTs)
    # all_GTs = all_GTs.transpose()
    all_GTs =all_GTs.applymap(convert_to_int)
    all_GTs_array= all_GTs.values
    all_GTs_array=all_GTs_array.astype(int)
    # pickle.dump(all_GTs_array, open('vcf_original_only_PASS_SNPs_only_GTs', 'wb'))

    # all_GTs_array =np.array(all_GTs)
    kf = KFold(n_splits=4)
    clf_tree = tree.DecisionTreeClassifier(criterion="entropy")
    score = 0
    total_conf_mat = np.zeros((2, 2))
    for train, test in kf.split(all_GTs_array):
        # test1=all_GTs_array[train, :]
        # test2= classification[train]
        clf_tree.fit(all_GTs_array[train, :], classification[train])
        score += clf_tree.score(all_GTs_array[test, :], classification[test])
        y_pred = clf_tree.predict(all_GTs_array[test, :])
        conf_mat = confusion_matrix(classification[test], y_pred)
        print(conf_mat)
        total_conf_mat += conf_mat

    avg_score = score / 4
    print(avg_score)
    print(total_conf_mat)
    # print(all_GTs)
