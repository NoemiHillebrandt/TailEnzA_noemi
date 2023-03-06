
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
from enzyme_prediction import *
from feature_generation import *
from Bio import pairwise2
file_for_prediction="Input/Testfile.fasta"
directory_of_classifiers="Classifier/all_enzyme_files_gomin5_gemin2_noemi/"
fastas_aligned_before=False
permutation_file="permutations.txt"
gbff_file = "gbff_files/streptomyces_coelicolor_a32.gbff"
output_directory="Output/"
filename_output=output_directory+file_for_prediction.split("/")[1].split(".")[0]+".csv"
enzymes=["p450","ycao","SAM","Methyl"]
BGC_types=["ripp","nrp","pk"]
include_charge_features=False

p450_dict = {}
methyl_dict = {}
radical_sam_dict = {}
ycao_dict= {}

p450_df = pd.DataFrame(columns=["sequence","produrct","cds_start","cds_end"])
ycao_df = pd.DataFrame(columns=["sequence","product","cds_start","cds_end"])
methyl_df = pd.DataFrame(columns=["sequence","produrct","cds_start","cds_end"])
radical_sam_df = pd.DataFrame(columns=["sequence","product","cds_start","cds_end"])


with open(permutation_file, 'r') as file:
    permutations = [line.rstrip('\n') for line in file]
# the model proteins are the proteins all proteins are aligned against (beforehand) . They must be within the training data named "reference"
model_proteins_for_alignment={"p450":"MSAVALPRVSGGHDEHGHLEEFRTDPIGLMQRVRDECGDVGTFQLAGKQVVLLSGSHANEFFFRAGDDDLDQAKAYPFMTPIFGEGVVFDASPERRKEMLHNAALRGEQMKGHAATIEDQVRRMIADWGEAGEIDLLDFFAELTIYTSSACLIGKKFRDQLDGRFAKLYHELERGTDPLAYVDPYLPIESLRRRDEARNGLVALVADIMNGRIANPPTDKSDRDMLDVLIAVKAETGTPRFSADEITGMFISMMFAGHHTSSGTASWTLIELMRHRDAYAAVIDELDELYGDGRSVSFHLRQIPQLENVLKETLRLHPPLIILMRVAKGEFEVQGHRIHEGDLVAASPAISNRIPEDFPDPHDFVPARYEQPRQEDLLNRWTWIPFGAGRHRCVGAAFAIMQIKAIFSVLLREYEFEMAQPPESYRNDHSKMVVQLAQPACVRYRRRTGV",
                              "ycao": "MDIKYKLASYRICSPEETFEKIQEALKKIETVEIKNIQHLDKVNIPVYYLKRRVVVDGKEGIAIHYGKGANDIQAKVSACMEAIERFSASYDKNKVKEKPDNPINVEDLILPQYADKNVKEWVEGIDIINNETIDVPADAVFYPTSGKLFRGNTNGLASGNNLDEAILHATLEIIERDAWSLADLARKIPTKINPEDAKNPLIHELIEKYEKAGVKIILKDLTSEFEIPVVAAISDDLSKNPLMLCVGVGCHLHPEIAILRALTEVAQSRASQLHGFRRDAKLREEFTSKIPYERLKRIHRKWFEFEGEINIADMPNNARYDLKKDLKFIKDKLSEFGFDKLIYVDLNKVGVDAVRVIIPKMEVYTIDRDRLSRRAFERVKKLYY",
                              "SAM": "MGSSHHHHHHSSGLVPRGSHMRTISEDILFRLEKFGGILINKTNFERIELDETEAFFLYLVQNHGIEIATSFFKKEIEMGKLERALSLNIYSDNNIEDSLNNPYETLQNARKHVAKLKKHNILSFPLELVIYPSMYCDLKCGFCFLANREDRNAKPAKDWERILRQAKDNGVLSVSILGGEPTRYFDIDNLLIACEELKIKTTITTNAQLIKKSTVEILAKSKYITPVLSLQTLDSKLNFELMGVRPDRQIKLAKYFNEVGKKCRINAVYTKQSYEQIIELVDFCIENKIDRFSVANYSEVTGYTKIKKKYDLADLRRLNEYVTDYITQREANLNFATEGCHLFTAYPELINNSIEFSEFDEMYYGCRAKYTKMEIMSNGDILPCIAFLGVNQTKQNAFEKDLLDVWYDDPLYGGIRSFRTKNSKCLSCGLLKICEGGCYVNLIKEKSPEYFRDSVCQL", 
                              #"Methyl": "MGSSHHHHHHSSGLVPRGSHMTTETTTATATAKIPAPATPYQEDIARYWNNEARPVNLRLGDVDGLYHHHYGIGPVDRAALGDPEHSEYEKKVIAELHRLESAQAEFLMDHLGQAGPDDTLVDAGCGRGGSMVMAHRRFGSRVEGVTLSAAQADFGNRRARELRIDDHVRSRVCNMLDTPFDKGAVTASWNNESTMYVDLHDLFSEHSRFLKVGGRYVTITGCWNPRYGQPSKWVSQINAHFECNIHSRREYLRAMADNRLVPHTIVDLTPDTLPYWELRATSSLVTGIEKAFIESYRDGSFQYVLIAADRV"
                              }
start=0
end=350
# the splitting list defines the functional fragments that the enzymes will be cut into
splitting_lists={"p450":[["begin",start,92],["sbr1",93,192],["sbr2",193,275],["core",276,395],["end",396,end],["fes1",54,115],["fes2",302,401]],
                 "ycao": [["begin",start,64],["sbr1",65,82],["f2",83,153],["sbr2",154,185],["f3",186,227],["sbr3",228,281],["f4",282,296],["sbr4",297,306],["f5",307,362],["sbr5",363,368],["end",369,end]],
                 "SAM": [["begin",start,106],["SAM",107,310],["bridging",311,346],["end",347,end]],
                 #"Methyl":[["begin",0,78],["SAM1",79,104],["SAM2",105,128],["SAM3",129,158],["SAM4",159,188],["SAM5",189,233],["end",234,255]]
                 }
fragments={"p450":["begin","sbr1","sbr2","core","end","fes1","fes2"],
           "ycao":["begin","sbr1","f2","sbr2","f3","sbr3","f4","sbr4","f5","sbr5","end"],
           "SAM":["begin","SAM","bridging","end"],
           #"Methyl":["begin","SAM1","SAM2","SAM3","SAM4","SAM5","end"]
           }
# Classifier, die für die jeweiligen Enzyme am besten balanced accuracy scores erzielt haben
classifiers = {"p450":"_AdaBoostClassifier_classifier.sav",
               "ycao":"_DecisionTreeClassifier_classifier.sav",
               "SAM":"_AdaBoostClassifier_classifier.sav",
               "Methyl":"_AdaBoostClassifier_classifier.sav"
               }

# aus den dataframes extrahieren, genbank extrakte, dataframes der einzelnen enzyme
for gb_record in SeqIO.parse(gbff_file, "genbank") :
    
    gb_feature = gb_record.features
    accession = gb_record.annotations['accessions'][0]+'.'+str(gb_record.annotations['sequence_version'])

    for i,feature in enumerate(gb_record.features):
        if feature.type=='CDS':
            if 'product' in feature.qualifiers: #verify it codes for a real protein (not pseudogene)
        
                #data in qualifiers are all lists, even if only 1 string, so [0] converts to string
                #use lower() to make sure weirdly capitilized strings get selected as well
                product=feature.qualifiers['product'][0].lower()

                if "radical sam" in product:
                    id = feature.qualifiers['locus_tag'][0]
                    sequence = feature.qualifiers['translation'][0]
                    products = feature.qualifiers['product'][0]
                    cds_start = int(feature.location.start)
                    if cds_start > 0 :
                        cds_start = cds_start + 1
                    cds_end = int(feature.location.end)
                    
                    radical_sam_dict[id] = {"sequence": sequence, "product": products, "cds_start": cds_start, "cds-end": cds_end}
                    radical_sam_df = pd.DataFrame(radical_sam_dict)
                    radical_sam_df = radical_sam_df.transpose()                                 

                elif "p450" in product:
                    id = feature.qualifiers['locus_tag'][0]
                    sequence = feature.qualifiers['translation'][0]
                    products = feature.qualifiers['product'][0]
                    cds_start = int(feature.location.start)
                    if cds_start > 0 :
                        cds_start = cds_start + 1
                    cds_end = int(feature.location.end)
                   
                    p450_dict[id] = {"sequence": sequence, "product": products, "cds_start": cds_start, "cds-end": cds_end}
                    p450_df = pd.DataFrame(p450_dict)
                    p450_df = p450_df.transpose()

                elif "methyltransferase" in product:    
                    id = feature.qualifiers['locus_tag'][0]
                    sequence = feature.qualifiers['translation'][0]
                    products = feature.qualifiers['product'][0]
                    cds_start = int(feature.location.start)
                    if cds_start > 0 :
                        cds_start = cds_start + 1
                    cds_end = int(feature.location.end)
                   
                    methyl_dict[id] = {"sequence": sequence, "product": products, "cds_start": cds_start, "cds-end": cds_end}
                    methyl_df = pd.DataFrame(methyl_dict)
                    methyl_df = methyl_df.transpose() 
                    #methyl_df(columns=['ID','Sequence','Product', 'CDS-Start', 'CDS-End'])  
                   

                elif "ycao" in product:
                    id = feature.qualifiers['locus_tag'][0]
                    sequence = feature.qualifiers['translation'][0]
                    products = feature.qualifiers['product'][0]
                    cds_start = int(feature.location.start)
                    if cds_start > 0 :
                        cds_start = cds_start + 1
                    cds_end = int(feature.location.end)
                    ycao_dict[id] = {"sequence": sequence, "product": products, "cds_start": cds_start, "cds-end": cds_end}
                    ycao_df = pd.DataFrame(ycao_dict)
                    ycao_df = ycao_df.transpose()
#print(methyl_df)
results = pd.DataFrame()
list_of_dataframes =  [ycao_df, methyl_df, p450_df, radical_sam_df]                
for dataframe in list_of_dataframes:    
#     list_results =[]
    for record in dataframe["sequence"]:
        print("hi")
#             # predict enzyme-type of given fasta
#             enzyme=enzyme_calculation(str(record.seq))[0] # obsolete
#             fragment_matrix=pd.DataFrame()
#             # align the AA sequence against the model protein for that enzyme type and fragment according to the splitting list
#             fewgaps = lambda x, y: -20 - y
#             specificgaps = lambda x, y: (-2 - y)
#             alignment = pairwise2.align.globalmc(model_proteins_for_alignment[enzyme], record.seq, 1, -1, fewgaps, specificgaps)
#             print(alignment)
#             fragment_matrix=fragment_alignment (alignment[0],splitting_lists[enzyme],fastas_aligned_before) # weil es das für jeweils einzelne Sequenzen macht, besteht die fragment matrix aus zwei Zeilen
#             print(fragment_matrix)
#             feature_matrix=featurize(fragment_matrix, permutations, fragments[enzyme], include_charge_features)
#             print (feature_matrix)
#             #select classifier
#             natural-product = classifier_np.predict(feature_matrix)
#             used_classifier=directory_of_classifiers+enzyme+ classifiers[enzyme] # fertig ergänzen für alle  Enzyme
#             classifier = pickle.load(open(used_classifier, 'rb'))
#             # predict substrate
#             predicted_BGC=classifier.predict(feature_matrix)
#             score_predicted_BGCs = classifier.predict_proba(feature_matrix)
#             list_result += classifier.predict_proba(feature_matrix) #an dfs anhängen

# # scoring:
# -> list of starts
# for start in starts:
#     #calcuklate frame
#     -> extarct enzyme directory_of_classifiers-> new df with only enzymes in BGC
#     -> scoring function
#     wenn scrore > wert:
#         extract genbank from source gb 
#         -> save
#         + append score to list