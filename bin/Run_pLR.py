#!/home/ionadmin/TaoY/src/miniconda2/bin/python
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
import os, sys, getopt, re, pickle

dpath = os.environ["DeeprimerPATH"]
binpath = dpath+"/bin"
datapath = dpath+"/data"
OBJpath = dpath+"/Trained_OBJ"
MIpath = dpath+"/Machine_input"
REPpath = dpath+"/Report"

sys.path.append(binpath)
import model_eval as mev

def main(argv):
	dpath = os.environ["DeeprimerPATH"]
	try:
		opts, args = getopt.getopt(argv,"hC:w:o:p:")
	except getopt.GetoptError:
		print('Run_pLR.py -C <penalty> -w <fit/pred/eval> -o <preprocessed_object> -p <pred_input>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print("[Command]: Run_randomforest.py -C <penalty> -w <fit/pred/eval> -o <preprocessed_object> -p <pred_input>" + "\n\n" +
					"Example:" + "\n\n" + 
					"1. Fit the model only:" + "\n\n" +
					"Run_pLR.py -C 1000 -w fit -o <sample_pre.fit.pickled>" + "\n\n" +
					"2. Evaluate the model:" + "\n\n" + 
					"Run_pLR.py -C 1000 -w eval -o <sample_split.eval.pickled>" + "\n\n" +
					"3. Make prediction:" + "\n\n" + 
					"Run_pLR.py -C 1000 -w pred -o <sample_pre.fit_pLR.machine> -p <pred_input>" + "\n")
			sys.exit()
		elif opt == "-C":
			pty = int(arg)
		elif opt == "-w":
			task = arg
		elif opt == "-o":
			obj = arg
		elif opt == "-p":
			pred_input = pd.read_csv(arg, delim_whitespace=True)
			pred = pred_input.values

	LW = 2
	RANDOM_STATE = 42
	tag = re.sub('\.pickled$', '', obj).split("/")[-1]

	if task == "fit":
		
		with open(obj, 'rb') as f:
                        pre_obj = pickle.load(f)
                        X_tr = pre_obj[0]
                        y_tr = pre_obj[1]
		if sum(y_tr == 1) >= 2*sum(y_tr == 0) or sum(y_tr == 1) <= (1/2)*sum(y_tr == 0):
			classifier = LogisticRegression(C=pty, penalty='l2', tol=0.01)
			sampler = RandomOverSampler(random_state=RANDOM_STATE)
			clf = make_pipeline(sampler, classifier)
		else:
			clf = LogisticRegression(C=pty, penalty='l2', tol=0.01)

		fitted = clf.fit(X_tr, y_tr)
		with open(OBJpath + "/" + tag + "_pLR.machine", 'wb') as f:
			pickle.dump(fitted, f, pickle.HIGHEST_PROTOCOL)

	elif task == "eval":

		with open(obj, 'rb') as f:
			pre_obj = pickle.load(f)
			X_tr = pre_obj[0]
			y_tr = pre_obj[1]
			X_te = pre_obj[2]
			y_te = pre_obj[3]
		
		if sum(y_tr == 1) >= 2*sum(y_tr == 0) or sum(y_tr == 1) <= (1/2)*sum(y_tr == 0):
			classifier = LogisticRegression(C=pty, penalty='l2', tol=0.01)
			sampler = RandomOverSampler(random_state=RANDOM_STATE)
			clf = make_pipeline(sampler, classifier)
		else:
			clf = LogisticRegression(C=pty, penalty='l2', tol=0.01)

		fitted = clf.fit(X_tr, y_tr)
		fpr, tpr, precision, recall, metrics = mev.performance_stat(X_te, y_te, fitted)
		mev.plot_eval(fpr, tpr, precision, recall, metrics, tag)
		with open(REPpath+"/"+tag+"_pLR.metrics", 'wb') as f:
			for key, value in metrics.items():
	                        f.write("%s %.3f" % (key, value)+'\n')
		
	
	elif task == "pred":
		tag1 = re.sub('\.machine$', '', obj).split("/")[-1]
		with open(obj, 'rb') as f:
			fitted_obj = pickle.load(f)
		pred_p = fitted_obj.predict_proba(pred)
		pred_c = fitted_obj.predict(pred)
		df = pd.DataFrame({"Probability_1": pred_p[:,1],
						"Probability_0": pred_p[:,0],
						"Predited_class": pred_c})
		df.to_csv(REPpath+"/"+tag1+".classification", index = None, sep = '\t')

	else:
		print("Not supported task. Choose from fit, pred, eval" + "\n")
		sys.exit()
	
if __name__ == "__main__":
	main(sys.argv[1:])


