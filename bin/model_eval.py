#!/home/ionadmin/TaoY/src/miniconda2/bin/python
import tensorflow as tf
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import matthews_corrcoef
import matplotlib
matplotlib.use('agg')
import pylab as plt
import os

#evaluation metrics
def performance_stat(X_te, y_te, fitted):

	pred_p = fitted.predict_proba(X_te)
	pred_c = fitted.predict(X_te)
	
	acc = sum(pred_c==y_te)/len(pred_c)
	fpr, tpr, thresholds = roc_curve(y_te, pred_p[:,1], pos_label=1)
	auROC = roc_auc_score(y_te, pred_p[:, 1])

	precision, recall, threshold = precision_recall_curve(y_te, pred_p[:,1])	
	f1 = metrics.f1_score(y_te, pred_c)
	auPRC = average_precision_score(y_te, pred_p[:,1])
	mcc = matthews_corrcoef(y_te, pred_c)
	metr = {"Accuracy": acc, "MCC": mcc, "F1": f1, "ROC_AUC": auROC, "PRC_AUC": auPRC}
	return fpr, tpr, precision, recall, metr

def performance_stat_nn(X_te, y_te, mach, meta):
	
	dpath = os.environ["DeeprimerPATH"]
	TFpath = dpath+"/TF_sessions/"+mach

	sess=tf.Session()
	saver = tf.train.import_meta_graph(TFpath+'/'+meta)
	#saver = tf.train.import_meta_graph("CNN-model.meta")
	saver.restore(sess, tf.train.latest_checkpoint(TFpath))
	
	graph = tf.get_default_graph()
	y_conv = graph.get_tensor_by_name("y_conv:0")
	y_ = graph.get_tensor_by_name("y_:0")
	x = graph.get_tensor_by_name("x:0")
	keep_prob = graph.get_tensor_by_name("keep_prob:0")

	X_te32 = X_te.astype(np.float32)
	Yte = np.column_stack((abs(1-y_te),y_te))

	feed_dict={x: X_te32, y_: Yte, keep_prob: 1.0}


	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	acc = accuracy.eval(feed_dict, session=sess)
	
	probs = tf.nn.softmax(y_conv)
	pred_p = sess.run(probs, feed_dict)

	prediction = tf.argmax(y_conv, 1)
	pred_c = prediction.eval(feed_dict, session=sess)

	fpr, tpr, thresholds = roc_curve(y_te, pred_p[:,1], pos_label=1)
	auROC = roc_auc_score(y_te, pred_p[:, 1])

	precision, recall, threshold = precision_recall_curve(y_te, pred_p[:,1])
	f1 = metrics.f1_score(y_te, pred_c)
	auPRC = average_precision_score(y_te, pred_p[:,1])
	mcc = matthews_corrcoef(y_te, pred_c)
	metr = {"Accuracy": acc, "MCC": mcc, "F1": f1, "ROC_AUC": auROC, "PRC_AUC": auPRC}
	return fpr, tpr, precision, recall, metr


def plot_eval(fpr, tpr, recall, precision, metr, tag):
	
	dpath = os.environ["DeeprimerPATH"]
	REPpath = dpath+"/Report"
	
	
	#plot ROC
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',
    	     lw=lw, label='ROC curve (area = %0.2f)' % metr['ROC_AUC'])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate', fontsize = 14)
	plt.ylabel('True Positive Rate', fontsize = 14)
	plt.title('Receiver operating characteristic curve', fontsize = 14)
	plt.legend(loc="lower right")
	plt.savefig(REPpath + "/ROC_" + tag + ".pdf", format = "pdf", dpi = 600)


	#plot precision - recall
	plt.figure()
	lw = 2
	plt.plot(recall, precision, color='darkorange',
    	     lw=lw, label='PRC curve (area = %0.2f)' % metr['PRC_AUC'])
	plt.plot([1, 0], [0.5, 0.5], color='grey', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('Recall', fontsize = 14)
	plt.ylabel('Precision', fontsize = 14)
	plt.title('Precision Recall curve', fontsize = 14)
	plt.legend(loc="lower right")
	plt.savefig(REPpath + "/Precsion-Recall_"+ tag +".pdf", format = "pdf", dpi = 600)
