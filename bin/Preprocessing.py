#!/home/ionadmin/TaoY/src/miniconda2/bin/python
import numpy as np
import pandas as pd
import os,sys,getopt,re
import warnings
import pickle

#sys.path.insert(0, 'C:\\ProgramData\\Anaconda3\\lib\\site-packages')
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#set path
dpath = os.environ["DeeprimerPATH"]
binpath = dpath+"/bin"
datapath = dpath+"/data"
OBJpath = dpath+"/Trained_OBJ"
MIpath = dpath+"/Machine_input"

sys.path.append(binpath)
import Kmers_map as km
import seq_cont as seqc

# format features and labels
def format_input(training, testing, n_kmers):
	
	#compute K-mers map
	kmer_tr = [None]*5
	kmer_te = [None]*5
	c = 0
	for i in [2,4,6,8,10]:
		kmer_tr[c] = km.comb_kmers(training, training, i, n_kmers)
		kmer_te[c] = km.comb_kmers(training, testing, i, n_kmers)
		c += 1

	kmer_tr = pd.concat(kmer_tr, axis = 1)
	kmer_te = pd.concat(kmer_te, axis = 1)

	#load seq_cont module
	seq_cont_tr = seqc.all_calc(training)
	seq_cont_te = seqc.all_calc(testing)

	#get the context data
	context_dep_tr = training.loc[:, 'fwdpvsp_mq':'revpvsi_a2']
	context_dep_te = testing.loc[:, 'fwdpvsp_mq':'revpvsi_a2']

	#fit rescaling
	fit_collection = rescaling_fit(kmer_tr, seq_cont_tr, context_dep_tr)

	#rescaling transform
	featured_rescl_tr = rescaling_transform(kmer_tr, seq_cont_tr, context_dep_tr, fit_collection)
	featured_rescl_te = rescaling_transform(kmer_te, seq_cont_te, context_dep_te, fit_collection)

	#combine all together
	ft_all_tr = np.column_stack((featured_rescl_tr, 
							training.loc[:,'fwdp.mean.map':'ins.min.map'].values))
	ft_all_te = np.column_stack((featured_rescl_te, 
							testing.loc[:,'fwdp.mean.map':'ins.min.map'].values))

	label_tr = training['class'].values
	label_te = testing['class'].values

	##Extract feature names
	Tm_gibs = ['fwdU.gibs.ave', 'fwdU.gibs.max', 'revU.gibs.ave', 
		'revU.gibs.max', 'fwdp.gibs.3p5.max', 'revp.gibs.3p5.max',
		'fwdp.Tm', 'revp.Tm', 'Delta.Tm', 'ins.gibs', 'ins.Tm']
	cnt_col = [col for col in seq_cont_tr.columns if col not in Tm_gibs]
	map_nam = ['fwdp.mean.map', 'fwd.min.map', 
				'revp.mean.map', 'revp.min.map',
				'ins.mean.map', 'ins.min.map']

	feat_names = list(kmer_tr.columns) + cnt_col + list(context_dep_tr) + Tm_gibs + map_nam

	return ft_all_tr, label_tr, ft_all_te, label_te, feat_names, fit_collection


# format features and labels
def training_obj(training, n_kmers):
	
	#usage:  X_tr, y_tr, X_name, fit_collection = training_obj(training, 40)

	#compute K-mers map
	kmer = [None]*5
	c = 0
	for i in [2,4,6,8,10]:
		kmer[c] = km.comb_kmers(training, training, i, n_kmers)
		c += 1

	kmer = pd.concat(kmer, axis = 1)

	#load seq_cont module
	seq_cont = seqc.all_calc(training)

	#get the context data
	context_dep = training.loc[:, 'fwdpvsp_mq':'revpvsi_a2']
	
	#fit rescaling
	fit_collection = rescaling_fit(kmer, seq_cont, context_dep)

	#rescaling transform
	featured_rescl = rescaling_transform(kmer, seq_cont, context_dep, fit_collection)

	#combine all together
	ft_all = np.column_stack((featured_rescl, 
							training.loc[:,'fwdp.mean.map':'ins.min.map'].values))

	label = training['class'].values

	##Extract feature names
	Tm_gibs = ['fwdU.gibs.ave', 'fwdU.gibs.max', 'revU.gibs.ave', 
		'revU.gibs.max', 'fwdp.gibs.3p5.max', 'revp.gibs.3p5.max',
		'fwdp.Tm', 'revp.Tm', 'Delta.Tm', 'ins.gibs', 'ins.Tm']
	cnt_col = [col for col in seq_cont.columns if col not in Tm_gibs]
	map_nam = ['fwdp.mean.map', 'fwd.min.map', 
				'revp.mean.map', 'revp.min.map',
				'ins.mean.map', 'ins.min.map']

	feat_names = list(kmer.columns) + cnt_col + list(context_dep) + Tm_gibs + map_nam

	return ft_all, label, feat_names, fit_collection, n_kmers


#rescaling
def rescaling_fit(kmer_tr, seq_cont, context_dep):

	#1. Imput NaN values
	imp = Imputer(missing_values='NaN', strategy='mean', axis=0).fit(seq_cont)
	seq_cont_imp = pd.DataFrame(imp.transform(seq_cont))
	seq_cont_imp.columns = seq_cont.columns

	#2. Rescaling features to the range of 0~1
	#Tm and Gibs will normalzied with L1
	Tm_gibs = ['fwdU.gibs.ave', 'fwdU.gibs.max', 'revU.gibs.ave', 
		'revU.gibs.max', 'fwdp.gibs.3p5.max', 'revp.gibs.3p5.max',
		'fwdp.Tm', 'revp.Tm', 'Delta.Tm', 'ins.gibs', 'ins.Tm']
	Thermo = seq_cont_imp[Tm_gibs]
	norml1 = preprocessing.Normalizer(norm='l1').fit(Thermo)

	#Counts will be scaled using MinMaxScaler;
	cnt_col = [col for col in seq_cont.columns if col not in Tm_gibs]
	Cnts = seq_cont_imp[cnt_col]

	mimx_Cnts = preprocessing.MinMaxScaler().fit(Cnts)
	mimx_Cxt = preprocessing.MinMaxScaler().fit(context_dep)
	mimx_kmers = preprocessing.MinMaxScaler().fit(kmer_tr)


	fit_collection = [imp, norml1, mimx_Cnts, mimx_Cxt, mimx_kmers]

	return fit_collection
	

def rescaling_transform(kmers, seq_cont, context_dep, fit_collection):

	seq_cont_imp = pd.DataFrame(fit_collection[0].transform(seq_cont))
	seq_cont_imp.columns = seq_cont.columns
	

	#Tm and Gibs will normalzied with L1
	Tm_gibs = ['fwdU.gibs.ave', 'fwdU.gibs.max', 'revU.gibs.ave', 
		'revU.gibs.max', 'fwdp.gibs.3p5.max', 'revp.gibs.3p5.max',
		'fwdp.Tm', 'revp.Tm', 'Delta.Tm', 'ins.gibs', 'ins.Tm']
	Thermo = seq_cont_imp[Tm_gibs]

	Thermo_norm = fit_collection[1].transform(Thermo)

	#Counts will be scaled using MinMaxScaler;
	cnt_col = [col for col in seq_cont.columns if col not in Tm_gibs]
	Cnts = seq_cont_imp[cnt_col]

	Cnts_minmax = fit_collection[2].transform(Cnts)
	Cxt_minmax = fit_collection[3].transform(context_dep)
	kmers_minmax = fit_collection[4].transform(kmers)

	#combine all together
	Feature_set = np.column_stack((kmers_minmax, Cnts_minmax, Cxt_minmax, Thermo_norm))
	return Feature_set


#format the prediciton data#
def format_input_pred(train_file, pred_file, pkd_obj):

	#Usage: X_pred, X_pred_nam = format_input_pred(training, testing, pkd_obj = 'objs_2class_e2e.pickle')

	#load picked training set
	#with open(pkd_obj, 'rb') as f:
	#	OBJ = pickle.load(f)

	fit_collection = pkd_obj[3]
	nk = pkd_obj[4]

	#compute K-mers map
	kmer = [None]*5
	c = 0
	for i in [2,4,6,8,10]:
		kmer[c] = km.comb_kmers(train_file, pred_file, i, nk)
		c += 1
	
	kmer = pd.concat(kmer, axis = 1)

	#load seq_cont module
	seq_cont = seqc.all_calc(pred_file)

	#get the context data
	context_dep = pred_file.loc[:, 'fwdpvsp_mq':'revpvsi_a2']

	#rescaling transform
	featured_rescl = rescaling_transform(kmer, seq_cont, context_dep, fit_collection)

	#combine all together
	ft_all = np.column_stack((featured_rescl, 
				pred_file.loc[:,'fwdp.mean.map':'ins.min.map'].values))

	##Extract feature names
	Tm_gibs = ['fwdU.gibs.ave', 'fwdU.gibs.max', 'revU.gibs.ave', 
		'revU.gibs.max', 'fwdp.gibs.3p5.max', 'revp.gibs.3p5.max',
		'fwdp.Tm', 'revp.Tm', 'Delta.Tm', 'ins.gibs', 'ins.Tm']
	cnt_col = [col for col in seq_cont.columns if col not in Tm_gibs]
	map_nam = ['fwdp.mean.map', 'fwd.min.map', 
				'revp.mean.map', 'revp.min.map',
				'ins.mean.map', 'ins.min.map']

	feat_names = list(kmer.columns) + cnt_col + list(context_dep) + Tm_gibs + map_nam

	return ft_all, feat_names


def main(argv):
	dpath = os.environ["DeeprimerPATH"]
	try:
		opts, args = getopt.getopt(argv,"ht:n:o:p:")
	except getopt.GetoptError:
		print('Preprocessing.py -t <train_set> -n <kmers> -o <trained_object> -p <pred_file>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print("[Command]: Preprocessing.py -t <train_set> -n <kmers> -o <trained_object> -p <pred_file>"+"\n"+
				"Examples:" + "\n\n" +
				"1. For model evaluation purpose, split the data into 2-fold, compute the features:"+"\n\n"+
				"Preprocessing.py -t <train_set> -n <kmers> -p split-eval" + "\n\n"+
				"2. Compute features for the training data only, save fitted object:" +"\n\n" +
				"Preprocessing.py -t <train_set> -n <kmers>" + "\n\n"+
				"3. Compute features for the prediction data according to the fitted training set:" + "\n\n" +
				"Preprocessing.py -t <train_set> -n <kmers> -o <trained_object> -p <pred_file>" + "\n")
			sys.exit()
		elif opt == "-p":
			pred_file = arg
		elif opt == "-t":
			train_set = arg
		elif opt == "-n":
			n_kmers = int(arg)
		elif opt == "-o":
			obj = arg

	train = pd.read_csv(train_set, delim_whitespace=True)
	out_nam = re.sub('\.txt$', '', train_set).split("/")[-1]

	# if pred_file isn't provided, the training set will be formated
	if 'pred_file' in locals():
		out_p_nam = re.sub('\.txt$', '', pred_file).split("/")[-1]
	if 'pred_file' not in locals():
		var = raw_input("Format training set could take long time. Do you want continue to [YES/NO]: ")
		if var == "YES":
			X_tr, y_tr, X_nam, fit_collection, n_kmers = training_obj(train, n_kmers)
			objnam = out_nam+"_pre.fit.pickled"
			with open(OBJpath+"/"+objnam, 'wb') as f:
				pickle.dump([X_tr, y_tr, X_nam, fit_collection, n_kmers], f, pickle.HIGHEST_PROTOCOL)
			print("Training data is formatted and placed in the Trained_OBJ folder.")
			
			df = pd.DataFrame(np.column_stack([X_tr, y_tr]))
			df.columns = X_nam+["class"]
			df.to_csv(MIpath+"/"+objnam + ".formatted", index = None, sep = '\t')
		else:
			sys.exit()
	
	elif pred_file == "split-eval":
		
		#split the data into training and testing
		training, testing = train_test_split(train, test_size=0.5, random_state=0)
		training.index = range(len(training))
		testing.index = range(len(testing))

		X_tr, y_tr, X_te, y_te, X_nam, fit_collection = format_input(training, testing, n_kmers)
		objnam = out_nam+"_split.eval.pickled"
		with open(OBJpath+"/"+objnam, 'wb') as f:
			pickle.dump([X_tr, y_tr, X_te, y_te, X_nam, fit_collection], f, pickle.HIGHEST_PROTOCOL)	

	else:
		pred = pd.read_csv(pred_file, delim_whitespace=True)
		with open(obj, 'rb') as f:
			fit_obj = pickle.load(f)
		X_pred, X_pred_nam = format_input_pred(train, pred, fit_obj)
		
		df = pd.DataFrame(X_pred)
		df.columns = X_pred_nam
		df.to_csv(MIpath+"/"+out_p_nam+".formatted", index = None, sep = '\t')

if __name__ == "__main__":
	main(sys.argv[1:])
