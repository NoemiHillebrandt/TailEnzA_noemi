import os
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.Align.Applications import MuscleCommandline
from Bio import AlignIO
from feature_generation import *
from Bio import pairwise2
import pickle

muscle = r"muscle"
#muscle = "/Users/noemihillebrandt/Documents/GoetheUni/Masterarbeit/main_project/TailEnzA/TailEnzA_original/Prediction/muscle5"

#file_for_prediction  ="Input/extract_rowfile.fasta"
directory_of_classifiers_BGC_type = "Classifier/muscle5_alignment_dataset/"
directory_of_classifiers_NP_affiliation = "Classifier/best_np_affiliation_classifier/"
fastas_aligned_before = True
permutation_file = "permutations.txt"
gbff_file = "gbff_files/streptomyces_coelicolor_a32.gbff"
output_directory = "Output_new/"
filename_output=output_directory+gbff_file.split("/")[1].split(".")[0]+".csv"
#enzymes=["p450","ycao","Methyl","SAM"]
enzymes=["p450","ycao"]
BGC_types=["ripp","nrp","pk"]
include_charge_features=True



with open(permutation_file, 'r') as file:
    permutations = [line.rstrip('\n') for line in file]
# the model proteins are the proteins all proteins are aligned against (beforehand) . They must be within the training data named "reference"
model_proteins_for_alignment={"p450":"MSAVALPRVSGGHDEHGHLEEFRTDPIGLMQRVRDECGDVGTFQLAGKQVVLLSGSHANEFFFRAGDDDLDQAKAYPFMTPIFGEGVVFDASPERRKEMLHNAALRGEQMKGHAATIEDQVRRMIADWGEAGEIDLLDFFAELTIYTSSACLIGKKFRDQLDGRFAKLYHELERGTDPLAYVDPYLPIESLRRRDEARNGLVALVADIMNGRIANPPTDKSDRDMLDVLIAVKAETGTPRFSADEITGMFISMMFAGHHTSSGTASWTLIELMRHRDAYAAVIDELDELYGDGRSVSFHLRQIPQLENVLKETLRLHPPLIILMRVAKGEFEVQGHRIHEGDLVAASPAISNRIPEDFPDPHDFVPARYEQPRQEDLLNRWTWIPFGAGRHRCVGAAFAIMQIKAIFSVLLREYEFEMAQPPESYRNDHSKMVVQLAQPACVRYRRRTGV",
                              "ycao": "MDIKYKLASYRICSPEETFEKIQEALKKIETVEIKNIQHLDKVNIPVYYLKRRVVVDGKEGIAIHYGKGANDIQAKVSACMEAIERFSASYDKNKVKEKPDNPINVEDLILPQYADKNVKEWVEGIDIINNETIDVPADAVFYPTSGKLFRGNTNGLASGNNLDEAILHATLEIIERDAWSLADLARKIPTKINPEDAKNPLIHELIEKYEKAGVKIILKDLTSEFEIPVVAAISDDLSKNPLMLCVGVGCHLHPEIAILRALTEVAQSRASQLHGFRRDAKLREEFTSKIPYERLKRIHRKWFEFEGEINIADMPNNARYDLKKDLKFIKDKLSEFGFDKLIYVDLNKVGVDAVRVIIPKMEVYTIDRDRLSRRAFERVKKLYY",
                              "SAM": "MGSSHHHHHHSSGLVPRGSHMRTISEDILFRLEKFGGILINKTNFERIELDETEAFFLYLVQNHGIEIATSFFKKEIEMGKLERALSLNIYSDNNIEDSLNNPYETLQNARKHVAKLKKHNILSFPLELVIYPSMYCDLKCGFCFLANREDRNAKPAKDWERILRQAKDNGVLSVSILGGEPTRYFDIDNLLIACEELKIKTTITTNAQLIKKSTVEILAKSKYITPVLSLQTLDSKLNFELMGVRPDRQIKLAKYFNEVGKKCRINAVYTKQSYEQIIELVDFCIENKIDRFSVANYSEVTGYTKIKKKYDLADLRRLNEYVTDYITQREANLNFATEGCHLFTAYPELINNSIEFSEFDEMYYGCRAKYTKMEIMSNGDILPCIAFLGVNQTKQNAFEKDLLDVWYDDPLYGGIRSFRTKNSKCLSCGLLKICEGGCYVNLIKEKSPEYFRDSVCQL", 
                              "Methyl": "MGSSHHHHHHSSGLVPRGSHMTTETTTATATAKIPAPATPYQEDIARYWNNEARPVNLRLGDVDGLYHHHYGIGPVDRAALGDPEHSEYEKKVIAELHRLESAQAEFLMDHLGQAGPDDTLVDAGCGRGGSMVMAHRRFGSRVEGVTLSAAQADFGNRRARELRIDDHVRSRVCNMLDTPFDKGAVTASWNNESTMYVDLHDLFSEHSRFLKVGGRYVTITGCWNPRYGQPSKWVSQINAHFECNIHSRREYLRAMADNRLVPHTIVDLTPDTLPYWELRATSSLVTGIEKAFIESYRDGSFQYVLIAADRV"
                              }
start=0
end=350
# the splitting list defines the functional fragments that the enzymes will be cut into
splitting_lists={"p450":[["begin",start,92],["sbr1",93,192],["sbr2",193,275],["core",276,395],["end",396,end],["fes1",54,115],["fes2",302,401]],
                 "ycao": [["begin",start,64],["sbr1",65,82],["f2",83,153],["sbr2",154,185],["f3",186,227],["sbr3",228,281],["f4",282,296],["sbr4",297,306],["f5",307,362],["sbr5",363,368],["end",369,end]],
                 "SAM": [["begin",start,106],["SAM",107,310],["bridging",311,346],["end",347,end]],
                 "Methyl":[["begin",0,78],["SAM1",79,104],["SAM2",105,128],["SAM3",129,158],["SAM4",159,188],["SAM5",189,233],["end",234,255]]
                 }
fragments={"p450":["begin","sbr1","sbr2","core","end","fes1","fes2"],
           "ycao":["begin","sbr1","f2","sbr2","f3","sbr3","f4","sbr4","f5","sbr5","end"],
           "SAM":["begin","SAM","bridging","end"],
           "Methyl":["begin","SAM1","SAM2","SAM3","SAM4","SAM5","end"]
           }
# Classifier, die für die jeweiligen Enzyme die besten balanced accuracy scores erzielt haben
classifiers_enzymes = {"p450":"_ExtraTreesClassifier_classifier.sav",
               "ycao":"_ExtraTreesClassifier_classifier.sav",
               "SAM":"_ExtraTreesClassifier_classifier.sav",
               "Methyl":"_ExtraTreesClassifier_classifier.sav"
               }

# classifier_NP_affiliation 
dict_classifier_NP_affiliation = {"p450":"_ExtraTreesClassifier_classifier.sav",
               "ycao":"_AdaBoostClassifier_classifier.sav",
               "SAM":"_AdaBoostClassifier_classifier.sav",
               "Methyl":"_ExtraTreesClassifier_classifier.sav"
              }

dict_parameters_alignment_BGC_affiliation = {"p450":[-8, -2],
               "ycao":[-8, -1],
               "SAM":[-2, -1],
               "Methyl":[-1, -1]
              }
def extract_feature_properties(feature):
    
    sequence = feature.qualifiers['translation'][0]
    products = feature.qualifiers['product'][0]
    cds_start = int(feature.location.start)
    if cds_start > 0 :
        cds_start = cds_start + 1
    cds_end = int(feature.location.end)
    return {"sequence": sequence, "product": products, "cds_start": cds_start, "cds_end": cds_end}
                    
# aus den complete_dataframes extrahieren, genbank extrakte, complete_dataframes der einzelnen enzyme
for gb_record in SeqIO.parse(gbff_file, "genbank"):
    gb_feature = gb_record.features
    #print(gb_feature)
    accession = gb_record.annotations['accessions'][0]+'.'+str(gb_record.annotations['sequence_version'])
    p450_dict = {}
    methyl_dict = {}
    radical_sam_dict = {}
    ycao_dict= {}
    for i,feature in enumerate(gb_record.features):
        if feature.type=='CDS':
            if 'product' in feature.qualifiers: #verify it codes for a real protein (not pseudogene)
        
                #data in qualifiers are all lists, even if only 1 string, so [0] converts to string
                #use lower() to make sure weirdly capitilized strings get selected as well
                product=feature.qualifiers['product'][0].lower()

                if "radical sam" in product:
                    try:
                        id = feature.qualifiers['locus_tag'][0]
                    except:
                        id = feature.qualifiers['protein_id'][0]
                    try:
                        radical_sam_dict[id] = extract_feature_properties(feature)  
                    except:
                        continue                                   

                elif "p450" in product:    
                    try:
                        id = feature.qualifiers['locus_tag'][0]
                    except:
                        id = feature.qualifiers['protein_id'][0]            
                    try:    
                        p450_dict[id] = extract_feature_properties(feature)
                    except:
                        continue

                elif "methyltransferase" in product:    
                    try:
                        id = feature.qualifiers['locus_tag'][0]
                    except:
                        id = feature.qualifiers['protein_id'][0]
                    try:
                        methyl_dict[id] = extract_feature_properties(feature)
                    except:
                        continue

                elif "ycao" in product:
                    try:
                        id = feature.qualifiers['locus_tag'][0]
                    except:
                        id = feature.qualifiers['protein_id'][0]
                    try:
                        ycao_dict[id] = extract_feature_properties(feature)     
                    except:
                        continue

    if radical_sam_dict:
        radical_sam_df = pd.DataFrame(radical_sam_dict)
        radical_sam_df = radical_sam_df.transpose()                    
        radical_sam_df.insert(0,"Enzyme","SAM")
    else:
        radical_sam_df = pd.DataFrame(columns=["sequence","product","cds_start","cds_end"])
        radical_sam_df.insert(0,"Enzyme","SAM")

    if p450_dict:
        p450_df = pd.DataFrame(p450_dict)
        p450_df = p450_df.transpose()
        p450_df.insert(0,"Enzyme","p450")                    
    else:
        p450_df = pd.DataFrame(columns=["sequence","product","cds_start","cds_end"])
        p450_df.insert(0,"Enzyme","p450") 

    if methyl_dict:
        methyl_df = pd.DataFrame(methyl_dict)
        methyl_df = methyl_df.transpose() 
        methyl_df.insert(0,"Enzyme","Methyl")
    else:
        methyl_df = pd.DataFrame(columns=["sequence","product","cds_start","cds_end"])
        methyl_df.insert(0,"Enzyme","Methyl")

    if ycao_dict:
        ycao_df = pd.DataFrame(ycao_dict)
        ycao_df = ycao_df.transpose()
        ycao_df.insert(0,"Enzyme","ycao")
    else:
        ycao_df = pd.DataFrame(columns=["sequence","product","cds_start","cds_end"])
        ycao_df.insert(0,"Enzyme","ycao")
        
    complete_dataframe = pd.concat([ycao_df, p450_df], axis=0)
    complete_dataframe["BGC_type"] = ""
    complete_dataframe["BGC_type_score"] = ""
    complete_dataframe["NP_BGC_affiliation"] = ""
    complete_dataframe["NP_BGC_affiliation_score"] = ""
    complete_dataframe["30kb_window_start"] = ""
    complete_dataframe["30kb_window_end"] = ""
    complete_dataframe["points_BGC_type"] = ""
    complete_dataframe["points_prob_value"] = ""
    #calcuklate frame and write to complete_dataframes
    complete_dataframe["30kb_window_start"] = complete_dataframe["cds_start"]
    complete_dataframe["30kb_window_start"] = complete_dataframe["30kb_window_start"].astype('int')
    complete_dataframe["30kb_window_end"] = complete_dataframe["cds_start"] + 30000
    complete_dataframe["30kb_window_end"] = complete_dataframe["30kb_window_end"].astype('int')

    #print(complete_dataframe)
    if len(complete_dataframe) != 0:
        fragment_rows=[]
        fragment_matrix = pd.DataFrame()
        enzyme = complete_dataframe["Enzyme"][0]
        translated_sequences = [SeqRecord(Seq(model_proteins_for_alignment[enzyme]), id ="Reference")]
        for index,row in complete_dataframe.iterrows():
            translated_sequences.append(SeqRecord(Seq(row["sequence"]), id = index))
        SeqIO.write(translated_sequences, "temp/temp_input.fasta", "fasta")
        [gap_opening_penalty, gap_extend_penalty] = dict_parameters_alignment_BGC_affiliation[enzyme]
        # command from coomand line in muscle 5
        muscle_commmand_line = f"{muscle} -align temp/temp_input.fasta -output temp/temp_output.fasta"

        os.system(muscle_commmand_line)
        alignment = AlignIO.read(open("temp/temp_output.fasta"), "fasta")

        # align the AA sequence against the model protein for that enzyme type and fragment according to the splitting list

        fragment_matrix = fragment_alignment(alignment,splitting_lists[enzyme],fastas_aligned_before)
        fragment_matrix.set_index(complete_dataframe.index)
        feature_matrix=featurize(fragment_matrix, permutations, fragments[enzyme], include_charge_features)
        # NP-BGC oder non NP-BGC Zuordnung
        natural_product_classifier = directory_of_classifiers_NP_affiliation+enzyme+dict_classifier_NP_affiliation[enzyme]
        classifier_NP_affiliation = pickle.load(open(natural_product_classifier, 'rb'))
        predicted_NP_affiliation = classifier_NP_affiliation.predict(feature_matrix)
        score_predicted_NP_affiliation = classifier_NP_affiliation.predict_proba(feature_matrix)
        complete_dataframe["NP_BGC_affiliation"] = predicted_NP_affiliation
        complete_dataframe["NP_BGC_affiliation_score"] = [max(score_list_NP_affiliation) for score_list_NP_affiliation in score_predicted_NP_affiliation] # finds highest value in matrix from 
        # BGC type Bestimmung
        BGC_type_classifier = directory_of_classifiers_BGC_type+enzyme+classifiers_enzymes[enzyme] # fertig ergänzen für alle  Enzyme
        classifier_BGC_type = pickle.load(open(BGC_type_classifier, 'rb'))
        # predict NP affiliation
        predicted_BGC = classifier_BGC_type.predict(feature_matrix)
        score_predicted_BGCs = classifier_BGC_type.predict_proba(feature_matrix)
        complete_dataframe["BGC_type"] = predicted_BGC
        complete_dataframe["BGC_type_score"] = [max(score_list_BGC_type) for score_list_BGC_type in score_predicted_BGCs] # checks for the highest score associated with the respective probability
        score_list_NP_affiliation = []
        score_list_BGC_type = []  
        # erstellt immer einen filtered dataframe aus jeder Zeile des complete dataframes, welche alle innerhalb des windows liegen
        for index, row in complete_dataframe.iterrows():
            window_start = row["30kb_window_start"]
            window_end = row["30kb_window_end"]
            filtered_dataframe = complete_dataframe[(complete_dataframe['cds_start'] >= window_start) & (complete_dataframe['cds_end'] <= window_end)]
            score = 0
            # gibt für jedes Fenster einen Score in Abhängigkeit von ripp-Zuordnung und von BGC-type Zuordnung
            for index, rows_rows in filtered_dataframe.iterrows():
                
                if rows_rows["BGC_type"]=="ripp":
                    score = score + 1 + rows_rows["BGC_type_score"]
                else:
                    score = score - rows_rows["BGC_type_score"] - 1
            # gibt Seq Record für jedes Fenster aus
            record = gb_record[window_start:window_end] 
            
            print(record)
# wenn scrore > wert:
#         extract genbank from source gb 
#         -> save
#         + append score to list

outpath = os.path.splitext(os.path.basename(genbank_path))[0] + ".fasta" 
SeqIO.write(intergenic_records, open(outpath,"w"), "fasta")

            #print(full_record)
for record in record: 
    SeqIO.write(record, "output_Genbank_file", "fasta")
    #print(filtered_dataframe[["Enzyme","BGC_type_score","BGC_type","30kb_window_start","30kb_window_end"]])
            
    