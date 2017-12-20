#!/home/ionadmin/TaoY/src/miniconda2/bin/python
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
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


def plot_confusion_matrix(cm, classes, normalize, title, tag):
    
    REPpath = dpath+"/Report"

    plt.figure()
    cmap = plt.cm.Blues
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(REPpath+"/"+tag+"_""_RF_confusion_m.pdf", format = "pdf", dpi = 600)


def main(argv):
	dpath = os.environ["DeeprimerPATH"]
	try:
		opts, args = getopt.getopt(argv,"ht:w:o:p:c:")
	except getopt.GetoptError:
		print('Run_randomforest.py -t <n_estimator> -c <n_class> -w <fit/pred/eval> -o <preprocessed_object> -p <pred_input>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print("[Command]: Run_randomforest.py -t <n_estimator> -w <fit/pred/eval> -o <preprocessed_object> -p <pred_input>" + "\n\n" +
					"Example:" + "\n\n" + 
					"1. Fit the model only:" + "\n\n" +
					"Run_randomforest.py -t 1000 -w fit -o <sample_pre.fit.pickled>" + "\n\n" +
					"2. Evaluate the model:" + "\n\n" + 
					"Run_randomforest.py -t 1000 -w eval -o <sample_split.eval.pickled>" + "\n\n" +
					"3. Make prediction:" + "\n\n" + 
					"Run_randomforest.py -t 1000 -w pred -o <sample_pre.fit_RF.machine> -p <pred_input>" + "\n")
			sys.exit()
		elif opt == "-t":
			n_est = int(arg)
		elif opt == "-w":
			task = arg
		elif opt == "-o":
			obj = arg
		elif opt == "-c":
			nclass = int(arg)
		elif opt == "-p":
			pred_input = pd.read_csv(arg, delim_whitespace=True)
			pred = pred_input.values
	
	if 'nclass' not in locals():
		nclass = 2
	LW = 2
	RANDOM_STATE = 42
	tag = re.sub('\.pickled$', '', obj).split("/")[-1]

	if task == "fit":
		
		with open(obj, 'rb') as f:
                        pre_obj = pickle.load(f)
                        X_tr = pre_obj[0]
                        y_tr = pre_obj[1]
		if sum(y_tr == 1) >= 2*sum(y_tr == 0) or sum(y_tr == 1) <= (1/2)*sum(y_tr == 0):
			classifier = RandomForestClassifier(n_estimators = n_est)
			sampler = RandomOverSampler(random_state=RANDOM_STATE)
			clf = make_pipeline(sampler, classifier)
		else:
			clf = RandomForestClassifier(n_estimators = n_est)

		fitted = clf.fit(X_tr, y_tr)
		if nclass == 2:
			with open(OBJpath + "/" + tag + "_RF.machine", 'wb') as f:
				pickle.dump(fitted, f, pickle.HIGHEST_PROTOCOL)
		elif nclass == 3:
			with open(OBJpath + "/" + tag + "_3C_RF.machine", 'wb') as f:
                                pickle.dump(fitted, f, pickle.HIGHEST_PROTOCOL)
		else:
			print("Not support class over 3. Try Regress." + "\n")
	                sys.exit()


	elif task == "eval":

		with open(obj, 'rb') as f:
			pre_obj = pickle.load(f)
			X_tr = pre_obj[0]
			y_tr = pre_obj[1]
			X_te = pre_obj[2]
			y_te = pre_obj[3]
		
		if sum(y_tr == 1) >= 2*sum(y_tr == 0) or sum(y_tr == 1) <= (1/2)*sum(y_tr == 0):
			classifier = RandomForestClassifier(n_estimators = n_est)
			sampler = RandomOverSampler(random_state=RANDOM_STATE)
			clf = make_pipeline(sampler, classifier)
		else:
			clf = RandomForestClassifier(n_estimators = n_est)

		fitted = clf.fit(X_tr, y_tr)
		if nclass == 2:
			fpr, tpr, precision, recall, metrics = mev.performance_stat(X_te, y_te, fitted)
			mev.plot_eval(fpr, tpr, precision, recall, metrics, tag)
			with open(REPpath+"/"+tag+"_RF.metrics", 'wb') as f:
				for key, value in metrics.items():
	            	            f.write("%s %.3f" % (key, value)+'\n')
	    	elif nclass == 3:
	    		pred_p = fitted.predict_proba(X_te)
			pred_c = fitted.predict(X_te)
			cnf_matrix = confusion_matrix(y_te, pred_c)
			np.set_printoptions(precision=2)
			class_names = ["0", "1", "2"]

			plot_confusion_matrix(cnf_matrix, class_names, True,'Normalized confusion matrix', tag)
			with open(REPpath+"/"+tag+"_3C_RF_eval.summay", 'wb') as f:
	            	            f.write(classification_report(y_te, pred_c, target_names=class_names))

	elif task == "pred":
		tag1 = re.sub('\.machine$', '', obj).split("/")[-1]
		with open(obj, 'rb') as f:
			fitted_obj = pickle.load(f)
		pred_p = fitted_obj.predict_proba(pred)
		pred_c = fitted_obj.predict(pred)
		
		if nclass == 2:
			df = pd.DataFrame({"Probability_1": pred_p[:,1],
							"Probability_0": pred_p[:,0],
							"Predited_class": pred_c})
			df.to_csv(REPpath+"/"+tag1+".2classification", index = None, sep = '\t')

		elif nclass == 3:
			df = pd.DataFrame({"Probability_2": pred_p[:,2],
							"Probability_1": pred_p[:,1],
							"Probability_0": pred_p[:,0],
							"Predited_class": pred_c})
			df.to_csv(REPpath+"/"+tag1+".3classification", index = None, sep = '\t')
		else:
			print("Not support class over 3. Try Regression.")
			sys.exit()

	else:
		print("Not supported task. Choose from fit, pred, eval" + "\n")
		sys.exit()
	
if __name__ == "__main__":
	main(sys.argv[1:])


