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
TF_CNN = TFpath + "/CNN"

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

#Deimensions: [batch_size, height, width, channel]
def conv1d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool1d(x):
    return tf.nn.max_pool(x, ksize=[1,1,2,1], strides=[1,1,2,1], padding='SAME')

def CNN_5layer(x, f):

    #Dimensions: 
    weights = {'W_conv1':tf.Variable(tf.truncated_normal([1,8,1,32], stddev = 0.1))
            ,'W_conv2':tf.Variable(tf.truncated_normal([1,8,32,32], stddev = 0.1))
            ,'W_conv3':tf.Variable(tf.truncated_normal([1,8,32,64], stddev = 0.1))
            ,'W_conv4':tf.Variable(tf.truncated_normal([1,8,64,64], stddev = 0.1))
            ,'W_fc':tf.Variable(tf.truncated_normal([104*64,1024], stddev = 0.1))
            ,'out':tf.Variable(tf.truncated_normal([1024,2], stddev = 0.1))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32]))
            ,'b_conv2':tf.Variable(tf.random_normal([32]))
            ,'b_conv3':tf.Variable(tf.random_normal([64]))
            ,'b_conv4':tf.Variable(tf.random_normal([64]))
            ,'b_fc':tf.Variable(tf.random_normal([1024]))
            ,'out':tf.Variable(tf.random_normal([2]))}

    x_ft = tf.reshape(x, shape=[-1,1,f,1])

    # conv layer caould be flexibly added here
    conv1 = conv1d(x_ft, weights['W_conv1']) + biases['b_conv1']
    conv2 = conv1d(conv1, weights['W_conv2']) + biases['b_conv2']
    conv2 = maxpool1d(conv1)
    conv3 = conv1d(conv2, weights['W_conv3']) + biases['b_conv3']
    conv4 = conv1d(conv3, weights['W_conv4']) + biases['b_conv4']
    conv4 = maxpool1d(conv4)

    pool_flat = tf.reshape(conv4, [-1, 104*64])
    fc1 = tf.nn.relu(tf.matmul(pool_flat, weights['W_fc']) + biases['b_fc'])

    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    fc1_drop = tf.nn.dropout(fc1, keep_prob)

    y_conv = tf.add(tf.matmul(fc1_drop, weights['out']), biases['out'], name = 'y_conv')

    return y_conv, keep_prob


def fit_CNN(X_tr, y_tr, n_f, dp, it, tag, desti):

    X_tr32 = X_tr.astype(np.float32)
    Ytr = np.column_stack((abs(1-y_tr),y_tr))

    x = tf.placeholder(tf.float32, shape = [None, n_f], name = 'x')
    y_ = tf.placeholder(tf.float32, shape = [None, 2], name = 'y_')
    y_conv, keep_prob = CNN_5layer(x, n_f)


    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(it):
        batch_xs, batch_ys = next_batch(50, X_tr32, Ytr)
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: dp}, session=sess)

    saver = tf.train.Saver()
    saver.save(sess, desti+"/"+tag+"_CNN.machine")



def main(argv):
    dpath = os.environ["DeeprimerPATH"]
    try:
        opts, args = getopt.getopt(argv,"hf:d:i:w:o:p:")
    except getopt.GetoptError:
        print('Run_CNN.py -f <n_feature> -d <drop_probability> -i <iterations> -w <fit/pred/eval> -o <preprocessed_object> -p <pred_input>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("[Command]: Run_CNN.py -f <n_feature> -d <drop_probability> -i <iterations> -w <fit/pred/eval> -o <preprocessed_object> -p <pred_input>" + "\n\n" +
                    "Example:" + "\n\n" + 
                    "1. Fit the model only:" + "\n\n" +
                    "Run_CNN.py -f 415 -d 0.5 -i <iterations> -w fit -o <sample_pre.fit.pickled>" + "\n\n" +
                    "2. Evaluate the model:" + "\n\n" + 
                    "Run_CNN.py -f 415 -d 0.5 -i <iterations> -w eval -o <sample_split.eval.pickled>" + "\n\n" +
                    "3. Make prediction:" + "\n\n" + 
                    "Run_CNN.py -f 415 -w pred -o <sample_pre.fit_CNN.machine.meta> -p <pred_input>" + "\n")
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
        
        fit_CNN(X_tr, y_tr, n_f, dp, it, tag, TF_CNN) #restores session in the TF_sessions folder

    elif task == "eval":

        with open(obj, 'rb') as f:
            pre_obj = pickle.load(f)
            X_tr = pre_obj[0]
            y_tr = pre_obj[1]
            X_te = pre_obj[2]
            y_te = pre_obj[3]
        
        fit_CNN(X_tr, y_tr, n_f, dp, it, tagi, TF_CNN)
        meta = tag+"_CNN.machine.meta"
        fpr, tpr, precision, recall, metrics = mev.performance_stat_nn(X_te, y_te, "CNN", meta)
        mev.plot_eval(fpr, tpr, precision, recall, metrics, tag)
        with open(REPpath+"/"+tag+"_CNN.metrics", 'wb') as f:
            	for key, value in metrics.items():
			f.write("%s %.3f" % (key, value)+'\n')

    elif task == "pred":
        meta = obj
	tag1 =  re.sub('\.machine\.meta$', '', obj).split("/")[-1]
        X_te32 = Xpred.astype(np.float32)
        sess=tf.Session()
        saver = tf.train.import_meta_graph(obj)
        saver.restore(sess, tf.train.latest_checkpoint(TF_CNN))
        

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


    




