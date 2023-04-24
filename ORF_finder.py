import argparse
import re
from Bio import SeqIO
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.Seq import Seq

# Parse command line arguments
parser = argparse.ArgumentParser(description="Annotate ORFs in intergenic regions of a GenBank file")
parser.add_argument("input_file", help="Input GenBank file")
args = parser.parse_args()

# Load the GenBank file
record = SeqIO.read(args.input_file, "genbank")
pattern = r'(.+)_([0-9]+)_([0-9]+)_([0-9]+\.[0-9]+)\.gb$'
matches = re.search(pattern, args.input_file)
window_start = int(matches.group(2))
window_end = int(matches.group(3))
print(window_start)
# Get the intergenic regions as SeqFeatures
intergenic_features = []
orfs = []
last_feature = None
for feature in record.features:
    if last_feature is not None and feature.location.start > last_feature.location.end:
        intergenic_loc = FeatureLocation(last_feature.location.end + 1, feature.location.start - 1)
        intergenic_features.append(SeqFeature(intergenic_loc, type="intergenic"))
    last_feature = feature

# Annotate ORFs and alternative start regions in the intergenic regions
for intergenic_feature in intergenic_features:
    intergenic_seq = intergenic_feature.extract(record.seq)
    for strand, nuc in [(+1, intergenic_seq), (-1, intergenic_seq.reverse_complement())]:
        for frame in range(3):
            trans = str(nuc[frame:].translate())
            trans_len = len(trans)
            aa_start = None
            for aa_end in range(0, trans_len):
                if trans[aa_end] == "M" or trans[aa_end] == "V":
                    aa_start = aa_end
                    orf_start = intergenic_feature.location.start + frame + aa_start * 3
                    if trans[aa_end] == "V":
                        alt_start = orf_start
                elif aa_start is not None and trans[aa_end] == "*":
                    orf_end = intergenic_feature.location.start + frame + aa_end * 3 + 2
                    orf_loc = FeatureLocation(orf_start, orf_end, strand=strand)
                    orf_feature = SeqFeature(orf_loc, type="ORF")
                    if not any(f.type == "CDS" and orf_loc.start >= f.location.start and orf_loc.end <= f.location.end for f in record.features):
                        record.features.append(orf_feature)
                        if aa_start == 0:
                            alt_start_loc = FeatureLocation(alt_start, alt_start + 2, strand=strand)
                            alt_start_feature = SeqFeature(alt_start_loc, type="alternative_start")
                            record.features.append(alt_start_feature)
                    aa_start = None
                elif aa_start is not None and aa_end == trans_len - 1:
                    orf_end = intergenic_feature.location.start + frame + aa_end * 3 + 2
                    orf_loc = FeatureLocation(orf_start, orf_end, strand=strand)
                    orf_feature = SeqFeature(orf_loc, type="ORF")
                    if not any(f.type == "CDS" and orf_loc.start >= f.location.start and orf_loc.end <= f.location.end for f in record.features):
                        record.features.append(orf_feature)
                        orfs.append(orf_feature)
                        if aa_start == 0:
                            alt_start_loc = FeatureLocation(alt_start, alt_start + 2, strand=strand)
                            alt_start_feature = SeqFeature(alt_start_loc, type="alternative_start")
                            record.features.append(alt_start_feature)
                    aa_start = None

# Write the updated GenBank file
output_file = args.input_file[:-6] + "_corefinder_output.gbk"
SeqIO.write(record, output_file, "genbank")
print(f"Output written to {output_file}")
output_file = args.input_file + "_corefinder_output.fasta"
with open(output_file, "w") as f:
    for orf in orfs:
        seq_id = f">{record.id}|{orf.location.start}|{orf.location.end}"
        #seq_id = f">{record.id}|{(orf.location.start+window_start)}|{orf.location.end+window_end}"

        print(seq_id)
        #f.write(f"{seq_id}\n{record[orf.location.start:orf.location.end].seq}\n")
