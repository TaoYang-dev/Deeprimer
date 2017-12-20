#!/usr/bin/python

import pandas as pd
import numpy as np
import collections as cl
import Tm_calc as Tmc
import os, sys, getopt

# Uprimer blocks, Gibs and Tm
def Ubl_gibs(seq):
	'''
	Input:
	seq: A U-replaced sequence. If no 'U' exists in the sequence,
	treat the sequence as a single block.

	Ouput: 
	Mean and maximum Gibs free energy arcoss blocks

	'''
	dna_conc = 0.00000025
	salt_conc = 0.05

	Ublocks = pd.Series(seq.split('U'))
	Ublocks = pd.Series([x for x in Ublocks if x != ''])
	gibs = Ublocks.apply(lambda x: Tmc.calc_tm_sl(x, dna_conc, salt_conc)['Gibs']/len(x))

	mgibs = gibs.mean()
	mxgibs = gibs.max()

	return {"ave_norm_gibs": mgibs, "max_norm_gibs": mxgibs}

#sequence content
def seq_cnt(seq):
	
	'''
	Input: a sequence
	Output:
	[ACTG]cnt: counts of [ACTG]
	GCp: the percent of G and C
	GCcnt_3p5: the count of G or C in the last 5 bases of 3' end
	'''
	Acnt = seq.count('A')
	Tcnt = seq.count('T')
	Ccnt = seq.count('C')
	Gcnt = seq.count('G')

	GCcnt_3p5 = seq[-5:].count('C') + seq[-5:].count('G')

	cnts = {"Acnt": Acnt, "Tcnt": Tcnt, "Ccnt": Ccnt, 
			"Gcnt": Ccnt, "GCcnt_3p5": GCcnt_3p5}

	return cnts

def all_calc(inputfile):
	
	dna_conc = 0.00000025
	salt_conc = 0.05

	fwdU_gibs = inputfile['fwdU.seq'].apply(lambda x: Ubl_gibs(x))
	revU_gibs = inputfile['revU.seq'].apply(lambda x: Ubl_gibs(x))

	fwdU_gibs_df = pd.DataFrame.from_records(fwdU_gibs)
	fwdU_gibs_df.columns = ["fwdU.gibs.ave", "fwdU.gibs.max"]

	revU_gibs_df = pd.DataFrame.from_records(revU_gibs)
	revU_gibs_df.columns = ["revU.gibs.ave", "revU.gibs.max"]

	fwdp_gibs_3p5 = inputfile['fwdp.seq'].apply(lambda x: Ubl_gibs(x[-5:])["max_norm_gibs"])
	fwdp_gibs_3p5.name ="fwdp.gibs.3p5.max"

	revp_gibs_3p5 = inputfile['revp.seq'].apply(lambda x: Ubl_gibs(x[-5:])["max_norm_gibs"])
	revp_gibs_3p5.name ="revp.gibs.3p5.max"


	fwdp_Tm = inputfile['fwdp.seq'].apply(lambda x: Tmc.calc_tm_sl(x, dna_conc, salt_conc)['Tm'])
	fwdp_Tm.name = "fwdp.Tm"

	revp_Tm = inputfile['revp.seq'].apply(lambda x: Tmc.calc_tm_sl(x, dna_conc, salt_conc)['Tm'])
	revp_Tm.name = "revp.Tm"

	delta_Tm = abs(fwdp_Tm - revp_Tm)
	delta_Tm.name = "Delta.Tm"

	gibs_ins = inputfile['insert.seq'].apply(lambda x: Tmc.calc_tm_sl(x, dna_conc, salt_conc)['Gibs']/len(x) if 'N' not in x else np.nan)
	gibs_ins.name = "ins.gibs"

	Tm_ins = inputfile['insert.seq'].apply(lambda x: Tmc.calc_tm_sl(x, dna_conc, salt_conc)['Tm'] if 'N' not in x else np.nan)
	Tm_ins.name = "ins.Tm"


	fwdp_cnts = inputfile['fwdp.seq'].apply(lambda x: seq_cnt(x))
	revp_cnts = inputfile['revp.seq'].apply(lambda x: seq_cnt(x))

	fwdp_cnts_df = pd.DataFrame.from_records(fwdp_cnts)
	fwdp_cnts_df.columns = ["fwdp.Acnt", "fwdp.Ccnt", "fwdp.GCcnt_3p5", "fwdp.Gcnt", "fwdp.Tcnt"]
	revp_cnts_df = pd.DataFrame.from_records(revp_cnts)
	revp_cnts_df.columns = ["revp.Acnt", "revp.Ccnt", "revp.GCcnt_3p5", "revp.Gcnt", "revp.Tcnt"]

	#Insert GC
	Ins_GC = inputfile['insert.seq'].apply(lambda x: x.count('G') + x.count('C'))
	Ins_GC.name = "Ins.GC"
	#print(Ins_GC.iloc[1:10,])

	# Lenth of fwdp, revp and insert
	fwdp_len = inputfile['fwdp.seq'].apply(lambda x: len(x))
	fwdp_len.name = "fwdp.len"
	revp_len = inputfile['revp.seq'].apply(lambda x: len(x))
	revp_len.name = "revp.len"
	ins_len = inputfile['insert.seq'].apply(lambda x: len(x))
	ins_len.name = "ins.len"

	All_feats = pd.concat([fwdp_len, revp_len, ins_len,
                fwdU_gibs_df, revU_gibs_df, fwdp_gibs_3p5, revp_gibs_3p5,
                fwdp_Tm, revp_Tm, delta_Tm, gibs_ins, Tm_ins,
                fwdp_cnts_df, revp_cnts_df, Ins_GC,
                inputfile['fwdNumTBlock'], inputfile['revNumTBlock'],
                inputfile['pass']], axis = 1)
	
	return All_feats




def main(argv):
	try:
		opts, args = getopt.getopt(argv,"hi:k:m:o:")
	except getopt.GetoptError:
		print('seq_cont.py -i <inputfile> -o <outputfile>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('seq_cont.py -i <inputfile> -o <outputfile>')
			sys.exit()
		elif opt == "-i":
			infile = arg
		elif opt == "-o":
			outfile = arg


	inputfile = pd.read_csv(infile, delim_whitespace=True)

	All_features = all_calc(inputfile)

	All_features.to_csv(outfile+"_conts.txt", index = None, na_rep = "NaN", sep = '\t')
	

	#outcome.to_csv(outfile+"_labels.txt", index = None, na_rep = "NaN", sep = '\t')

if __name__ == "__main__":
	main(sys.argv[1:])
