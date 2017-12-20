#!/usr/bin/python

import pandas as pd
import numpy as np
import collections as cl
import operator as op
import os, sys, getopt, itertools

# function to extract k-mers

def find_kmers(string, k):
    
      kmers = []
      n = len(string)

      for i in range(0, n-k+1):
           kmers.append(string[i:i+k])

      return kmers

def count_kmer(seqs, kmers_set, k, seq_spec):

	kmers = seqs.apply(lambda x: find_kmers(x.upper(), k))

	kmers_sel = kmers.apply(lambda x: [y for y in x if y in kmers_set])

	#count the numbers of discovered motif in training
	zero_set = np.zeros((len(seqs), len(kmers_set)))

	df_zeros = pd.DataFrame(zero_set, columns=kmers_set, dtype = int)

	kmers_cnts = kmers_sel.apply(lambda x: dict(cl.Counter(x)))

	for i in range(len(df_zeros)):
		df_zeros.loc[df_zeros.index[i],list(kmers_cnts[i])] = kmers_cnts[i]

	Counts_set = df_zeros

	Counts_set.columns = [str(col) + '_' + seq_spec for col in Counts_set.columns]

	return Counts_set

# function to extract the k_mers set
def kmers_rf(seqs_tr, k, num, seq_spec):
	
	#Discover kmers in training set
	kmers_tr = seqs_tr.apply(lambda x: find_kmers(x.upper(), k))

	kmers_all = list(kmers_tr)
	combined = list(itertools.chain.from_iterable(kmers_all))
	kmers_set = dict(cl.Counter(combined))

	#select the top 'num' kmers by total counts
	kmers_set_sort = sorted(kmers_set.items(), key=op.itemgetter(1))

	if (len(kmers_set) > int(num)):
		idx = int(num)
	else:
		idx = len(kmers_set)

	kmers_set_sel =  dict(kmers_set_sort[-idx:])

	return list(kmers_set_sel)

def comb_kmers(infile_tr, infile_te, k, num):

	specs = {"F": "fwdp.seq", "R":"revp.seq", "I":"insert.seq"}
	
	train_set = [None]*3
	test_set = [None]*3

	c = 0
	for i in list(specs):
		if c < 2:
			kmers_set = kmers_rf(infile_tr[specs[i]], k, int(num/2), i)
			test_set[c] = count_kmer(infile_te[specs[i]], kmers_set, k, i)
		else:
			kmers_set = kmers_rf(infile_tr[specs[i]], k, num, i)
			test_set[c] = count_kmer(infile_te[specs[i]], kmers_set, k, i)	
		c += 1

	All_kmers_te = pd.concat(test_set, axis = 1)
	return All_kmers_te


def main(argv):
	try:
		opts, args = getopt.getopt(argv,"hi:k:m:o:")
	except getopt.GetoptError:
		print('Kmers_map.py -i <inputfile> -k <number-kmers> -m <nmotif> -o <outputfile>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('Kmers_map.py -i <inputfile> -k <number-kmers> -m <nmotif> -o <outputfile>')
			sys.exit()
		elif opt == "-i":
			inputfile = arg
		elif opt == "-k":
			n_mers = int(arg)
		elif opt == "-m":
			nmotif = float(arg)
		elif opt == "-o":
			outputfile = arg
	
	inp = pd.read_csv(inputfile, delim_whitespace=True)

	kmers = comb_kmers(inp, inp, n_mers, nmotif) 
	kmers.to_csv(outputfile + ".txt", index = None, sep = '\t')

if __name__ == "__main__":
	main(sys.argv[1:])

