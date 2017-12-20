#!/home/ionadmin/TaoY/src/miniconda2/bin/python
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import pylab as plt
from sklearn.ensemble import ExtraTreesClassifier
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import pickle, math, sys, getopt, re, os

dpath = os.environ["DeeprimerPATH"]
binpath = dpath+"/bin"
datapath = dpath+"/data"
OBJpath = dpath+"/Trained_OBJ"
MIpath = dpath+"/Machine_input"
REPpath = dpath+"/Report"

# Build a forest and compute the feature importances
def forest_importance(X, y, X_nam, n_est, r, tag):
	
	forest = ExtraTreesClassifier(n_estimators=n_est, random_state=0)

	forest.fit(X, y)
	importances = forest.feature_importances_
	indices = np.argsort(importances)[::-1]
	
	#output top r ranked featureds
	with open(REPpath +'/' + tag+'_topranks_'+str(r)+'.txt', 'w') as f:
		f.write("%s" % "Feature ranking:\n")
		for l in range(r):
			f.write("%d. %s" % (l + 1, X_nam[indices[l]]) + '\n')

	return importances, indices

# Plot the feature importances of the forest
def plot_imp_rank(importances, indices, X_nam, tag):
	plt.figure(figsize=(14,8))
	plt.title("Feature importances")
	plt.bar(range(100), importances[indices[0:100]],
    		color="blue", orientation='vertical')
	plt.xlim([-1, 100])
	plt.ylim([0, 1.6*max(importances[indices[0:50]])])

	plt.figtext(0.18, 0.85, "Feature importance ranking:", fontsize=14, fontweight='bold')
	for i in range(50):
		xc = 0.18 + (i//10)*0.14
		yc = 0.82 - (i%10)*0.03
		ranks = str(i+1)+". "+X_nam[indices[i]]
		plt.figtext(xc,yc,ranks, fontsize = 12)
	plt.figtext(0.7, 0.4, "Suffix\n_I: Insert\n_F: Forward primer\n_R: Reverse primer", color = "orange", fontsize=14)
	plt.savefig(REPpath+"/"+tag+"_important_ranks.pdf", format = "pdf", dpi = 600)

#cluster the features and make the dendrogram
def feature_cluster(X, X_nam, indices, r, tag):
	
	a = X[:,indices[0:r]]
	dist100 = pdist(a.T, metric = 'correlation')
	Z = linkage(dist100)

	fig = plt.figure(figsize=(12,15*math.ceil(r/50)))
	ax = fig.add_subplot(1, 1, 1)

	l = [X_nam[indices[i]] for i in range(100)]
	figure = dendrogram(Z, labels = l, leaf_font_size=12, orientation='right',
						color_threshold = 0.3)

	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_visible(False)
	plt.xlabel('Distance: correlation', fontsize = 14)
	ax.get_xaxis().tick_bottom()
	ax.spines['bottom'].set_position(('outward', 10))
	plt.savefig(REPpath+"/"+tag+"_"+str(r)+"_feature_clusters.pdf", format = "pdf", dpi = 600)

def main(argv):
	try:
		opts, args = getopt.getopt(argv,"ht:r:o:")
	except getopt.GetoptError:
		print('Get_importance.py -t <n_estimator> -r <top_n_ranks> -o <preprocessed_object>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print("[Command]: Get_importance.py -t <n_estimator> -r <top_n_ranks> -o <preprocessed_object>" + "\n")
			sys.exit()
		elif opt == "-t":
			n_est = int(arg)
		elif opt == "-r":
			r = int(arg)
		elif opt == "-o":
			obj = arg

	LW = 2
	tag = re.sub('\.pickled$', '', obj).split("/")[-1]

	with open(obj, 'rb') as f:
        	pre_obj = pickle.load(f)
        	X = pre_obj[0]
        	y = pre_obj[1]
       		X_nam = pre_obj[2]

		importance, indices = forest_importance(X, y, X_nam, n_est, r, tag)
		plot_imp_rank(importance, indices, X_nam, tag)
		feature_cluster(X, X_nam, indices, r, tag)

if __name__ == "__main__":
	main(sys.argv[1:])
