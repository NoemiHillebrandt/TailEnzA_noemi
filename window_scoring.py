from sys import displayhook
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
#from enzyme_prediction import *
from feature_generation import *
from Bio import pairwise2
from prediciton_module import *
import pickle

#file_for_prediction  ="Input/extract_rowfile.fasta"
directory_of_classifiers_BGC_type = "Classifier/best_adaboost_classifier/"
directory_of_classifiers_NP_affiliation = "Classifier/classifiers_NP_BGC_affiliation/"
fastas_aligned_before = False
permutation_file = "permutations.txt"
#gbff_file = "gbff_files/cutout_streptomyces_coelicolor_a32.gbff"
output_directory = "Output/"
#filename_output=output_directory+file_for_prediction.split("/")[1].split(".")[0]+".csv"
enzymes=["p450","ycao"]
BGC_types=["ripp","nrp","pk"]
include_charge_features=True

# scoring part
for dataframe in list_of_dataframes:
    
    #dataframe.columns.get_loc("")
    dataframe["30kb_window_start"] = ""
    dataframe["30kb_window_end"] = ""
    #calcuklate frame and write to dataframes
    dataframe["30kb_window_start"] = dataframe["cds_start"]
    dataframe["30kb_window_start"] = dataframe["30kb_window_start"].astype('int')
    dataframe["30kb_window_end"] = dataframe["cds_start"] + 30000
    dataframe["30kb_window_end"] = dataframe["30kb_window_end"].astype('int')
    #print(dataframe[["cds_start","cds_end","30kb_window_start","30kb_window_end"]])
    for index, row in dataframe.iterrows():
        print(dataframe)
        #print(row["30kb_window_end"])
        window_start = row["30kb_window_start"]
        window_end = row["30kb_window_end"]
        filtered_dataframe = dataframe[(dataframe['cds_start'] >= window_start) & (dataframe['cds_end'] <= window_end)]
        filtered_dataframe["points"] = ""
        filtered_dataframe["points"] = [1 if x == "ripp" else -1 for x in filtered_dataframe["BGC_type"]]
        #filtered_dataframe["points"] = [1 if y >= 0.32 else -1 for y in filtered_dataframe["BGC_type_score"]]
        print(filtered_dataframe[["Enzyme","BGC_type_score","BGC_type","points","30kb_window_start","30kb_window_end"]])
        #dataframe["score"] = ""
        #dataframe["score"] = row["points"].sum()

    # score = scoring(filtered_df)
    # score in tabelle
    # wenn score > wert:
    #     -> extract genbank

    # for window_start in dataframe["30kb_window_start"].iterrows():
    #     if dataframe.loc[dataframe["cds_start"] >= dataframe["30kb_window_start"], "30kb_window_start"]:
    #         print("hi")
    

   # window = gb_record.seq[str(dataframe["30kb_window_start"]):str(dataframe["30kb_window_end"])]

        


# #     -> extarct enzyme directory_of_classifiers-> new df with only enzymes in BGC
#     -> scoring function
#     wenn scrore > wert:
#         extract genbank from source gb 
#         -> save
#         + append score to list