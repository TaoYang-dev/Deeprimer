#!/home/ionadmin/TaoY/src/miniconda2/bin/python
import tensorflow as tf
import numpy as np
import pandas as pd
import os, sys, getopt, re, pickle
import warnings
warnings.filterwarnings("ignore")

#set path
dpath = os.environ["DeeprimerPATH"]
binpath = dpath+"/bin"
datapath = dpath+"/data"
OBJpath = dpath+"/Trained_OBJ"
MIpath = dpath+"/Machine_input"
TFpath = dpath+"/TF_sessions"
REPpath = dpath+"/Report"
TF_FNN = TFpath+"/FNN"

sys.path.append(binpath)
import model_eval as mev

#Define next_batch function
def next_batch(num, data, labels):
    
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.random_normal(shape=shape)
    return tf.Variable(initial)

##################Wrapping a 6 layer fully connected neural network##################
# Create model
def fnn_6layer(x, n_node, n_class, n_feature):

    # Store layers weight & bias
    weights = {
        'h1': weight_variable([n_feature, n_node]),
        'h2': weight_variable([n_node, n_node]),
        'h3': weight_variable([n_node, n_node]),
        'h4': weight_variable([n_node, n_node]),
        'h5': weight_variable([n_node, n_node]),
        'h6': weight_variable([n_node, n_node]),
        'out': weight_variable([n_node, n_class])
        }
    biases = {
        'b1': bias_variable([n_node]),
        'b2': bias_variable([n_node]),
        'b3': bias_variable([n_node]),
        'b4': bias_variable([n_node]),
        'b5': bias_variable([n_node]),
        'b6': bias_variable([n_node]),
        'out': bias_variable([n_class])
        }

    #x_drop = tf.nn.dropout(x, keep_prob=keep_prob_input)
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

    # Layer 1
    layer_1 = tf.nn.relu(tf.matmul(x, weights['h1']) + biases['b1'])
    layer_1_dr = tf.nn.dropout(layer_1, keep_prob)
    # Layer 2
    layer_2 = tf.nn.relu(tf.matmul(layer_1_dr, weights['h2']) + biases['b2'])
    layer_2_dr = tf.nn.dropout(layer_2, keep_prob)
    # Layer 3
    layer_3 = tf.nn.relu(tf.matmul(layer_2_dr, weights['h3']) + biases['b3'])
    layer_3_dr = tf.nn.dropout(layer_3, keep_prob)
    # Layer 4
    layer_4 = tf.nn.relu(tf.matmul(layer_3_dr, weights['h4']) + biases['b4'])
    layer_4_dr = tf.nn.dropout(layer_4, keep_prob)
    # Layer 5
    layer_5 = tf.nn.relu(tf.matmul(layer_4_dr, weights['h5']) + biases['b5'])
    layer_5_dr = tf.nn.dropout(layer_5, keep_prob)
    # Layer 6
    layer_6 = tf.nn.relu(tf.matmul(layer_5_dr, weights['h6']) + biases['b6'])
    layer_6_dr = tf.nn.dropout(layer_6, keep_prob)
    # Output layer with linear activation
    y_conv = tf.add(tf.matmul(layer_6_dr, weights['out']), biases['out'], name = 'y_conv')
    return y_conv, keep_prob


def fit_FNN(X_tr, y_tr, n_f, dp, it, tag, desti):

    X_tr32 = X_tr.astype(np.float32)
    Ytr = np.column_stack((abs(1-y_tr),y_tr))

    x = tf.placeholder(tf.float32, shape = [None, n_f], name = 'x')
    y_ = tf.placeholder(tf.float32, shape = [None, 2], name = 'y_')
    y_conv, keep_prob = fnn_6layer(x, 1200, 2, n_f)
    
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(it):
        batch_xs_tr, batch_ys_tr = next_batch(50, X_tr32, Ytr)
        sess.run(train_step, feed_dict={x: batch_xs_tr, y_: batch_ys_tr, keep_prob: dp})

    saver = tf.train.Saver()
    saver.save(sess, desti+"/"+tag+"_FNN.machine")


def main(argv):
    dpath = os.environ["DeeprimerPATH"]
    try:
        opts, args = getopt.getopt(argv,"hf:d:i:w:o:p:")
    except getopt.GetoptError:
        print('Run_FNN.py -f <n_feature> -i <iterations> -d <drop_probability> -w <fit/pred/eval> -o <preprocessed_object> -p <pred_input>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("[Command]: Run_FNN.py -f <n_feature> -d <drop_probability> -i <iterations> -w <fit/pred/eval> -o <preprocessed_object> -p <pred_input>" + "\n\n" +
                    "Example:" + "\n\n" + 
                    "1. Fit the model only:" + "\n\n" +
                    "Run_FNN.py -f 415 -d 0.5 -i <iterations> -w fit -o <sample_pre.fit.pickled>" + "\n\n" +
                    "2. Evaluate the model:" + "\n\n" + 
                    "Run_FNN.py -f 415 -d 0.5 -i <iterations> -w eval -o <sample_split.eval.pickled>" + "\n\n" +
                    "3. Make prediction:" + "\n\n" + 
                    "Run_FNN.py -f 415 -w pred -o <sample_pre.fit_FNN.machine.meta> -p <pred_input>" + "\n")
            sys.exit()
        elif opt == "-f":
            n_f = int(arg)
        elif opt == "-d":
            dp = float(arg)
	elif opt == "-i":
	    it = int(arg)
        elif opt == "-w":
            task = arg
        elif opt == "-o":
            obj = arg
        elif opt == "-p":
            pred_input = pd.read_csv(arg, delim_whitespace=True)
            Xpred = pred_input.values 
    
    LW = 2
    tag = re.sub('\.pickled$', '', obj).split("/")[-1]

    if task == "fit":
        
        with open(obj, 'rb') as f:
            pre_obj = pickle.load(f)
        X_tr = pre_obj[0]
        y_tr = pre_obj[1]
        
        fit_FNN(X_tr, y_tr, n_f, dp, it, tag, TF_FNN) #restores session in the TF_sessions folder

    elif task == "eval":

        with open(obj, 'rb') as f:
            pre_obj = pickle.load(f)
            X_tr = pre_obj[0]
            y_tr = pre_obj[1]
            X_te = pre_obj[2]
            y_te = pre_obj[3]
        
        fit_FNN(X_tr, y_tr, n_f, dp, it, tag, TF_FNN)
        meta = tag+"_FNN.machine.meta"
        fpr, tpr, precision, recall, metrics = mev.performance_stat_nn(X_te, y_te, 'FNN', meta)
        mev.plot_eval(fpr, tpr, precision, recall, metrics, tag)
        with open(REPpath+"/"+tag+"_FNN.metrics", 'wb') as f:
		for key, value in metrics.items():
                        f.write("%s %.3f" % (key, value)+'\n')

    elif task == "pred":
        meta = obj
        tag1 =  re.sub('\.machine\.meta$', '', obj).split("/")[-1]
        X_te32 = Xpred.astype(np.float32)
        sess=tf.Session()
        saver = tf.train.import_meta_graph(obj)
        saver.restore(sess, tf.train.latest_checkpoint(TF_FNN))
        

        graph = tf.get_default_graph()
        y_conv = graph.get_tensor_by_name("y_conv:0")
        y_ = graph.get_tensor_by_name("y_:0")
        x = graph.get_tensor_by_name("x:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        feed_dict={x: X_te32, keep_prob: 1.0}

        probs = tf.nn.softmax(y_conv)
        pred_p = sess.run(probs, feed_dict)

        prediction = tf.argmax(y_conv,1)
        pred_c = prediction.eval(feed_dict, session=sess)

        df = pd.DataFrame({"Probability_1": pred_p[:,1],
                        "Probability_0": pred_p[:,0],
                        "Predited_class": pred_c})
        df.to_csv(REPpath+"/"+tag1+".classification", index = None, sep = '\t')

    else:
        print("Not supported task. Choose from fit, pred, eval" + "\n")
        sys.exit()
    
if __name__ == "__main__":
    main(sys.argv[1:])




