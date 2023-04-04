import os
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.Align.Applications import MuscleCommandline
from Bio import AlignIO
#from enzyme_prediction import *
from feature_generation import *
from Bio import pairwise2
import pickle

muscle = r"muscle"
#muscle = "/home/noemi/Documents/muscle3"
#file_for_prediction  ="Input/extract_rowfile.fasta"
directory_of_classifiers_BGC_type = "Classifier/trained_machine_learning_classifiers_repeated/"
directory_of_classifiers_NP_affiliation = "Classifier/best_np_affiliation_classifier/"
fastas_aligned_before = True
permutation_file = "permutations.txt"
gbff_file = "gbff_files/streptomyces_venezuelae.gbff"
output_directory = "Output_new/"
filename_output=output_directory+gbff_file.split("/")[1].split(".")[0]+".csv"
#enzymes=["p450","ycao","Methyl","SAM"]
enzymes=["p450","ycao","SAM","Methyl"]
BGC_types=["ripp","nrp","pk"]
include_charge_features=True

p450_dict = {}
methyl_dict = {}
radical_sam_dict = {}
ycao_dict= {}

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
                    
# aus den dataframes extrahieren, genbank extrakte, dataframes der einzelnen enzyme
for gb_record in SeqIO.parse(gbff_file, "genbank"):
    gb_feature = gb_record.features
    accession = gb_record.annotations['accessions'][0]+'.'+str(gb_record.annotations['sequence_version'])

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
                    radical_sam_dict[id] = extract_feature_properties(feature)                                 

                elif "p450" in product:    
                    try:
                        id = feature.qualifiers['locus_tag'][0]
                    except:
                        id = feature.qualifiers['protein_id'][0]            
                    p450_dict[id] = extract_feature_properties(feature)
                    

                elif "methyltransferase" in product:    
                    try:
                        id = feature.qualifiers['locus_tag'][0]
                    except:
                        id = feature.qualifiers['protein_id'][0]
                    methyl_dict[id] = extract_feature_properties(feature)

                elif "ycao" in product:
                    try:
                        id = feature.qualifiers['locus_tag'][0]
                    except:
                        id = feature.qualifiers['protein_id'][0]
                    ycao_dict[id] = extract_feature_properties(feature)                 
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
    


#list_of_dataframes = [methyl_df]
list_of_dataframes = [ycao_df,radical_sam_df,methyl_df,p450_df]     
results = pd.DataFrame()
for dataframe in list_of_dataframes: 
    dataframe["BGC_type"] = ""
    dataframe["BGC_type_score"] = ""
    dataframe["NP_BGC_affiliation"] = ""
    dataframe["NP_BGC_affiliation_score"] = ""
    #print(dataframe)
    if len(dataframe) == 0:
        continue
    fragment_rows=[]
    fragment_matrix = pd.DataFrame()
    enzyme = dataframe["Enzyme"][0]
    translated_sequences = [SeqRecord(Seq(model_proteins_for_alignment[enzyme]), id ="Reference")]
    #list_results =[]
    for index,row in dataframe.iterrows():
        #print(enzyme)
        translated_sequences.append(SeqRecord(Seq(row["sequence"]), id = index))
    SeqIO.write(translated_sequences, "temp/temp_input.fasta", "fasta")
    [gap_opening_penalty, gap_extend_penalty] = dict_parameters_alignment_BGC_affiliation[enzyme]
    
    # command muscle3 from comand line
    #muscle_commmand_line = f"{muscle} -in temp/temp_input.fasta -out temp/temp_output.fasta -gapopen {gap_opening_penalty} -gapextend {gap_extend_penalty} -center 0 -seqtype protein"
    #command muscle5 from command line
    muscle_commmand_line = f"{muscle} -align temp/temp_input.fasta -output temp/temp_output.fasta"
    print(muscle_commmand_line, type(muscle_commmand_line))
    os.system(muscle_commmand_line)
    alignment = AlignIO.read(open("temp/temp_output.fasta"), "fasta")

#       align the AA sequence against the model protein for that enzyme type and fragment according to the splitting list

    fragment_matrix = fragment_alignment(alignment,splitting_lists[enzyme],fastas_aligned_before)
    #print(fragment_matrix, dataframe)
    fragment_matrix.set_index(dataframe.index)
    #print(fragment_matrix)
    feature_matrix=featurize(fragment_matrix, permutations, fragments[enzyme], include_charge_features)
    #print (feature_matrix)
    # NP-BGC oder non NP-BGC Zuordnung
    natural_product_classifier = directory_of_classifiers_NP_affiliation+enzyme+dict_classifier_NP_affiliation[enzyme]
    classifier_NP_affiliation = pickle.load(open(natural_product_classifier, 'rb'))
    predicted_NP_affiliation = classifier_NP_affiliation.predict(feature_matrix)
    score_predicted_NP_affiliation = classifier_NP_affiliation.predict_proba(feature_matrix)
    dataframe["NP_BGC_affiliation"] = predicted_NP_affiliation
    dataframe["NP_BGC_affiliation_score"] = score_predicted_NP_affiliation
    # BGC type Bestimmung
    BGC_type_classifier = directory_of_classifiers_BGC_type+enzyme+classifiers_enzymes[enzyme] # fertig ergänzen für alle  Enzyme
    classifier_BGC_type = pickle.load(open(BGC_type_classifier, 'rb'))
    # predict substrate
    predicted_BGC = classifier_BGC_type.predict(feature_matrix)
    score_predicted_BGCs = classifier_BGC_type.predict_proba(feature_matrix)
    results= pd.DataFrame(score_predicted_BGCs, columns=classifier_BGC_type.classes_)

    dataframe["BGC_type"] = predicted_BGC
    dataframe["BGC_type_score"] = [max(score_list) for score_list in score_predicted_BGCs]
    #print(dataframe)
    print(dataframe.head(60)[["Enzyme","cds_start","cds_end","NP_BGC_affiliation","NP_BGC_affiliation_score","BGC_type","BGC_type_score"]])
