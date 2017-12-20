#!/home/ionadmin/TaoY/src/miniconda2/bin/python
import tensorflow as tf
import numpy as np
from scipy import interp
import matplotlib
matplotlib.use('agg')
import pylab as plt
from itertools import cycle
from scipy import interp
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from scipy.interpolate import griddata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import Run_CNN as CNN
import Run_randomforest as RF
import Run_SVM as SVM
import Run_FNN as FNN
import Run_pLR as pLR
import model_eval as mev
import pickle, os, sys, getopt, re


dpath = os.environ["DeeprimerPATH"]
binpath = dpath+"/bin"
datapath = dpath+"/data"
OBJpath = dpath+"/Trained_OBJ"
MIpath = dpath+"/Machine_input"
REPpath = dpath+"/Report"
TFpath = dpath+"/TF_sessions"
TF_CNN = TFpath+"/CNN"
TF_FNN = TFpath+"/FNN"
TF_TEMP = TFpath+"/TEMP"

#cross validation
def get_cv_scores(X, y, cla, k, tag):
	
	random_state = np.random.RandomState(42)
	cv = StratifiedKFold(n_splits=k)

	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)
	mean_precision = 0.0
	mean_recall = np.linspace(0, 1, 100)
	fpr = [None]*k
	tpr = [None]*k
	precision = [None]*k
	recall = [None]*k
	roc_auc = [None]*k
	auPRC = [None]*k
	metrics = [None]*k

	lw = 2
	
	if cla == "CNN" or cla == "FNN":
		n_f = raw_input("Tell the machine your number of features: ")
                dp = raw_input("Node dropping probability (put 0.5 if you don't know): ")
                it = raw_input("Number of iteractions (put 10000 if you don't know): ")
	elif cla == "RF":
		n_est = raw_input("number of estimators (put 1000 if you don't know): ")
	elif cla == "SVM" or cla == "pLR":
		pty = raw_input("Penalty (put 1000 if you don't know): ")
	else:
		print("Pick one classifier from CNN, FNN, RF, SVM, pLR")
		sys.exit()
	
	i = 0
	for train, test in cv.split(X, y):
    	
    		if cla == "CNN":
    			tf.reset_default_graph()
			CNN.fit_CNN(X[train], y[train], int(n_f), float(dp), int(it), tag, TF_TEMP)
			meta = tag+"_CNN.machine.meta"
			fpr[i], tpr[i], precision[i], recall[i], metrics[i] = mev.performance_stat_nn(X[test], y[test], 'CNN', meta)
		elif cla == "FNN":
			tf.reset_default_graph()
			FNN.fit_FNN(X[train], y[train], int(n_f), float(dp), int(it), tag, TF_TEMP)
			meta = tag+"_FNN.machine.meta"
			fpr[i], tpr[i], precision[i], recall[i], metrics[i] = mev.performance_stat_nn(X[test], y[test], 'FNN', meta)
		else:
			if cla == "RF":
				clf = RandomForestClassifier(n_estimators=int(n_est))
			elif cla == "SVM":
				clf = svm.SVC(kernel='rbf', probability = True, C = int(pty))
			elif cla == "pLR":
				clf = LogisticRegression(C=int(pty), penalty='l2', tol=0.01)
			else:
				print("Pick one classifier from CNN, FNN, RF, SVM, pLR")
				sys.exit()
			fitted = clf.fit(X[train], y[train])
			fpr[i], tpr[i], precision[i], recall[i], metrics[i] = mev.performance_stat(X[test], y[test], fitted)

    		mean_tpr += interp(mean_fpr, fpr[i], tpr[i])
    		mean_tpr[0] = 0.0
    		roc_auc[i] = metrics[i]['ROC_AUC']
    
    		mean_precision += griddata(recall[i], precision[i], mean_recall)
    		auPRC[i] = metrics[i]['PRC_AUC']
    		i += 1

	mean_tpr /= cv.get_n_splits(X, y)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)

	mean_precision /= cv.get_n_splits(X, y)
	mean_auPRC = auc(mean_recall, mean_precision)

	scores = [fpr, tpr, precision, recall, roc_auc, auPRC]
	means = [mean_fpr, mean_tpr, mean_precision, mean_recall, mean_auc, mean_auPRC]
	return scores, means

#make plot
def make_cv_plot(k, scores, means, tag, mach):
	
	lw = 2
	fpr, tpr, precision, recall, roc_auc, auPRC = scores
	mean_fpr, mean_tpr, mean_precision, mean_recall, mean_auc, mean_auPRC = means
	
	for i in range(k):
		fig1 = plt.figure(1)
		if i == 0:
			ax1 = fig1.add_subplot(1, 1, 1)
		plt.plot(fpr[i], tpr[i], lw=lw, label='ROC fold %d (area = %0.2f)' % (i, roc_auc[i]))
		fig2 = plt.figure(2)
		if i == 0:
			ax2 = fig2.add_subplot(1, 1, 1)
		plt.plot(recall[i], precision[i], lw=lw, label='PRC fold %d (area = %0.2f)' % (i, auPRC[i]))

   	plt.figure(1)
	plt.plot(mean_fpr, mean_tpr, color='m', linestyle='--',
				label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
	plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
				label='Luck')
	ax1.spines['top'].set_visible(False)
	ax1.spines['right'].set_visible(False)
	ax1.get_xaxis().tick_bottom()
	ax1.get_yaxis().tick_left()
	ax1.spines['left'].set_position(('outward', 10))
	ax1.spines['bottom'].set_position(('outward', 10))
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate', fontsize = 12)
	plt.ylabel('True Positive Rate', fontsize = 12)
	plt.title('Receiver operating characteristic', fontsize = 12)
	plt.legend(loc="lower right")
	plt.savefig(REPpath+"/"+tag+"_"+mach+"_ROC_CV.pdf", format = "pdf", dpi = 600)

	#Precision Recall
	plt.figure(2)
	plt.plot(mean_recall, mean_precision, color='m', linestyle='--',
					label='Mean PRC (area = %0.2f)' % mean_auPRC, lw=lw)
	plt.plot([1, 0], [0.5, 0.5], linestyle='--', lw=lw, color='k', label='Luck')
	ax2.spines['top'].set_visible(False)
	ax2.spines['right'].set_visible(False)
	ax2.get_xaxis().tick_bottom()
	ax2.get_yaxis().tick_left()
	ax2.spines['left'].set_position(('outward', 10))
	ax2.spines['bottom'].set_position(('outward', 10))
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('Recall', fontsize = 12)
	plt.ylabel('Precision', fontsize = 12)
	plt.title('Precision Recall curve', fontsize = 12)
	plt.legend(loc="lower left")
	plt.savefig(REPpath+"/"+tag+"_"+mach+"_PRC_CV.pdf", format = "pdf", dpi = 600)


def compare_all(X_tr, y_tr, X_te, y_te, n_f, dp, it, pty, n_est, tag):

	fpr = [None]*5
	tpr = [None]*5
	precision = [None]*5
	recall = [None]*5
	roc_auc = [None]*5
	auPRC = [None]*5
	metrics = [None]*5
	
	#Run_CNN
	tf.reset_default_graph()
	CNN.fit_CNN(X_tr, y_tr, int(n_f), float(dp), int(it), tag, TF_TEMP)
	meta = tag+"_CNN.machine.meta"
	fpr[0], tpr[0], precision[0], recall[0], metrics[0] = mev.performance_stat_nn(X_te, y_te, 'TEMP', meta)
	roc_auc[0] = metrics[0]['ROC_AUC']; auPRC[0] = metrics[0]['PRC_AUC']
	
	#Run FNN
	tf.reset_default_graph()
	FNN.fit_FNN(X_tr, y_tr, int(n_f), float(dp), int(it), tag, TF_TEMP)
	meta = tag+"_FNN.machine.meta"
	fpr[1], tpr[1], precision[1], recall[1], metrics[1] = mev.performance_stat_nn(X_te, y_te, 'TEMP', meta)
	roc_auc[1] = metrics[1]['ROC_AUC']; auPRC[1] = metrics[1]['PRC_AUC']
	
	#Run RF
	clf = RandomForestClassifier(n_estimators=int(n_est))
	fitted = clf.fit(X_tr, y_tr)
	fpr[2], tpr[2], precision[2], recall[2], metrics[2] = mev.performance_stat(X_te, y_te, fitted)
	roc_auc[2] = metrics[2]['ROC_AUC']; auPRC[2] = metrics[2]['PRC_AUC']
	
	#Run SVM
	clf = svm.SVC(kernel='rbf', probability = True, C = int(pty))
	fitted = clf.fit(X_tr, y_tr)
	fpr[3], tpr[3], precision[3], recall[3], metrics[3] = mev.performance_stat(X_te, y_te, fitted)
	roc_auc[3] = metrics[3]['ROC_AUC']; auPRC[3] = metrics[3]['PRC_AUC']
	
	#Run pLR
	clf = LogisticRegression(C=int(pty), penalty='l2', tol=0.01)
	fitted = clf.fit(X_tr, y_tr)
	fpr[4], tpr[4], precision[4], recall[4], metrics[4] = mev.performance_stat(X_te, y_te, fitted)
	roc_auc[4] = metrics[4]['ROC_AUC']; auPRC[4] = metrics[4]['PRC_AUC']

	cla_nam = ['CNN', 'FNN', 'RF', 'SVM', 'pLR']

	ind = np.argmax(auPRC)

	return fpr, tpr, precision, recall, roc_auc, auPRC, cla_nam[ind]

def plot_benchmark(fpr, tpr, precision, recall, roc_auc, auPRC, best, tag):
	lw = 2
	cla_nam = ['CNN', 'FNN', 'RF', 'SVM', 'pLR']

	for i in range(5):
		fig1 = plt.figure(1)
		if i == 0:
			ax1 = fig1.add_subplot(1, 1, 1)
		plt.plot(fpr[i], tpr[i], lw=lw, label='%s (area = %0.2f)' % (cla_nam[i], roc_auc[i]))
		fig2 = plt.figure(2)
		if i == 0:
			ax2 = fig2.add_subplot(1, 1, 1)
		plt.plot(recall[i], precision[i], lw=lw, label='%s (area = %0.2f)' % (cla_nam[i], roc_auc[i]))

	#ROC
   	plt.figure(1)
	plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
				label='Luck')
	ax1.spines['top'].set_visible(False)
	ax1.spines['right'].set_visible(False)
	ax1.get_xaxis().tick_bottom()
	ax1.get_yaxis().tick_left()
	ax1.spines['left'].set_position(('outward', 10))
	ax1.spines['bottom'].set_position(('outward', 10))
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate', fontsize = 12)
	plt.ylabel('True Positive Rate', fontsize = 12)
	plt.title('ROC - Best player: ' + best , fontsize = 12)
	plt.legend(loc="lower right")
	plt.savefig(REPpath+"/"+tag+"_ROC_benchmark.pdf", format = "pdf", dpi = 600)

	#Precision Recall
	plt.figure(2)
	plt.plot([1, 0], [0.5, 0.5], linestyle='--', lw=lw, color='k', label='Luck')
	ax2.spines['top'].set_visible(False)
	ax2.spines['right'].set_visible(False)
	ax2.get_xaxis().tick_bottom()
	ax2.get_yaxis().tick_left()
	ax2.spines['left'].set_position(('outward', 10))
	ax2.spines['bottom'].set_position(('outward', 10))
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('Recall', fontsize = 12)
	plt.ylabel('Precision', fontsize = 12)
	plt.title('Precision-Recall - Best player: ' + best, fontsize = 12)
	plt.legend(loc="lower left")
	plt.savefig(REPpath+"/"+tag+"_PRC_bechmark.pdf", format = "pdf", dpi = 600)


def main(argv):
    dpath = os.environ["DeeprimerPATH"]
    try:
        opts, args = getopt.getopt(argv,"hM:k:f:d:C:n:i:w:o:p:")
    except getopt.GetoptError:
        print('Run_deeprimer.py -M <machine> -f <n_feature> -i <iterations> -d <drop_probability> -C <penalty> \
        				 -n <RF_n_estimator> -w <bm/cv/pred> -o <preprocessed_object> -p <pred_input>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("[Command]: Run_deeprimer.py -M <machine> -k <cross-validation fold> -f <n_feature> -i <iterations> -d <drop_probability> -C <penalty> \
-n <RF_n_estimator> -w <bm/cv/pred> -o <preprocessed_object> -p <pred_input> " + "\n\n" + 
        				 "1. Benchmark classifiers" + "\n\n" +
        				 "Run_deeprimer.py -w bm -f 415 -i 10000 -d 0.5 -C 1000 -n 1000 -o <sample_split.eval.pickled>" + "\n\n" +
        				 "2. Cross-validate user-specified machine" + "\n\n" + "" + "Random Forest as an example:" + "\n" +
        				 "Run_deeprimer.py -w cv -M RF -k 3 -o <sample_pre.fit.pickled>" + "\n\n" + 
        				 "3. Predict with the best performance machine" + "\n\n" + "Random Forest as an example:" + "\n" +
        				 "Run_deeprimer.py -w pred -M RF -n 1000 -o <sample_pre.fit.machine>" + "\n")
            sys.exit()
        elif opt == "-M":
            cla_nam = arg
        elif opt == "-k":
            k = int(arg)
        elif opt == "-f":
            n_f = int(arg)
        elif opt == "-d":
            dp = float(arg)
	elif opt == "-i":
	    it = int(arg)
	elif opt == "-n":
	    n_est = int(arg)
	elif opt == "-C":
	    pty = int(arg)
        elif opt == "-w":
            task = arg
        elif opt == "-o":
            obj = arg
        elif opt == "-p":
            pred_input = arg
            Xpd = pd.read_csv(arg, delim_whitespace=True)
            Xpred = Xpd.values 
    
    LW = 2
    tag = re.sub('\.pickled$', '', obj).split("/")[-1]

    if task == "bm":
    	with open(obj, 'rb') as f:
    		data = pickle.load(f)
    	X_tr = data[0]
    	y_tr = data[1]
	X_te = data[2]
	y_te = data[3]

    	fpr, tpr, precision, recall, roc_auc, auPRC, best \
	= compare_all(X_tr, y_tr, X_te, y_te, n_f, dp, it, pty, n_est, tag)
    	plot_benchmark(fpr, tpr, precision, recall, roc_auc, auPRC, best, tag)
	os.system('rm -f ' + TF_TEMP + '/*')
    elif task == "cv":
    	with open(obj, 'rb') as f:
    		data = pickle.load(f)
    	X = data[0]
    	y = data[1]

    	scores, means = get_cv_scores(X, y, cla_nam, k, tag)
    	make_cv_plot(k, scores, means, tag, cla_nam)
	os.system('rm -f ' + TF_TEMP + '/*')

    elif task == "pred":
    	if cla_nam == "RF":
    		os.system("Run_randomforest.py -t" + str(n_est) +" -w pred -o " + str(obj) + " -p " + str(pred_input))
    	elif cla_nam == "CNN":
    		os.system("Run_CNN.py -f " + str(n_f) + " -w pred -o " + obj + " -p " + pred_input)
    	elif cla_nam == "FNN":
    		os.system("Run_FNN.py -f " + str(n_f) + " -w pred -o " + obj + " -p " + pred_input)
    	elif cla_nam == "SVM":
    		os.system("Run_SVM.py -C " + str(pty + " -w pred -o " + obj + " -p " + pred_input)
    	elif cla_nam == "pLR":
    		os.system("Run_pLR.py -C " + str(pty) + " -w pred -o " + obj + " -p " + pred_input)
    	else:
    		print("This machine has not been added to Deeprimer! Pick one from RF, SVM, pLR, CNN, FNN")
    		sys.exit()
    else:
        print("Not supported task. Choose from bm, cv, pred" + "\n")
        sys.exit()
    
if __name__ == "__main__":
    main(sys.argv[1:])

