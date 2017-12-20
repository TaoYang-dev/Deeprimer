#!/home/ionadmin/TaoY/src/miniconda2/bin/python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

import matplotlib
matplotlib.use('agg')
import pylab as plt
import numpy as np
import pandas as pd
import os, sys, getopt, re, pickle, itertools

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
		opts, args = getopt.getopt(argv,"ht:w:o:p:")
	except getopt.GetoptError:
		print('Run_RF_regressor.py -t <n_estimator> -w <fit/pred/eval> -o <preprocessed_object> -p <pred_input>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print("[Command]: Run_RF_regressor.py -t <n_estimator> -w <fit/pred/eval> -o <preprocessed_object> -p <pred_input>" + "\n\n" +
					"Example:" + "\n\n" + 
					"1. Fit the model only:" + "\n\n" +
					"Run_RF_regressor.py -t 1000 -w fit -o <sample_pre.fit.pickled>" + "\n\n" +
					"2. Evaluate the model:" + "\n\n" + 
					"Run_RF_regressor.py -t 1000 -w eval -o <sample_split.eval.pickled>" + "\n\n" +
					"3. Make prediction:" + "\n\n" + 
					"Run_RF_regressor.py -t 1000 -w pred -o <sample_pre.fit_RF.machine> -p <pred_input>" + "\n")
			sys.exit()
		elif opt == "-t":
			n_est = int(arg)
		elif opt == "-w":
			task = arg
		elif opt == "-o":
			obj = arg
		elif opt == "-p":
			pred_input = pd.read_csv(arg, delim_whitespace=True)
			pred = pred_input.values

	LW = 2
	tag = re.sub('\.pickled$', '', obj).split("/")[-1]

	if task == "fit":
		
		with open(obj, 'rb') as f:
                        pre_obj = pickle.load(f)
                        X_tr = pre_obj[0]
                        y_tr = pre_obj[1]

		X, y = make_regression(n_features=4, n_informative=2,
                       random_state=42, shuffle=False)
		regr = RandomForestRegressor(max_depth=2, random_state=0)
		regr.fit(X, y)

		fitted = clf.fit(X_tr, y_tr)
		with open(OBJpath + "/" + tag + "_RF.machine", 'wb') as f:
				pickle.dump(fitted, f, pickle.HIGHEST_PROTOCOL)

		with open(OBJpath + "/" + tag + "_3C_RF.machine", 'wb') as f:
                pickle.dump(fitted, f, pickle.HIGHEST_PROTOCOL)

	elif task == "eval":
		with open(obj, 'rb') as f:
			pre_obj = pickle.load(f)
			X_tr = pre_obj[0]
			y_tr = pre_obj[1]
			X_te = pre_obj[2]
			y_te = pre_obj[3]
		
		clf = RandomForestClassifier(n_estimators = n_est)

		fitted = clf.fit(X_tr, y_tr)
		fpr, tpr, precision, recall, metrics = mev.performance_stat(X_te, y_te, fitted)
		mev.plot_eval(fpr, tpr, precision, recall, metrics, tag)
		with open(REPpath+"/"+tag+"_RF.metrics", 'wb') as f:
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
		df.to_csv(REPpath+"/"+tag1+".2classification", index = None, sep = '\t')

	else:
		print("Not supported task. Choose from fit, pred, eval" + "\n")
		sys.exit()
	
if __name__ == "__main__":
	main(sys.argv[1:])


