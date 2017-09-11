from __future__ import division, print_function, absolute_import

import time
import os
import sys
import re
import csv
import codecs
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
import tensorflow as tf
import glob
from datetime import datetime
from filelogger import FileLogger
from scipy import misc

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize, rescale, rotate, setup, warp, AffineTransform

sys.path.append(os.path.abspath('..'))
#reload(sys)
#sys.setdefaultencoding('utf-8')

BASE_DIR = './input/'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
IMAGE_DIR= BASE_DIR + 'images/' 
IMAGE_EXT= ".jpg"
batch_size = 1
MARGIN_OFFSET= 64 
SHAPE_OFFSET = 128
TEXTURE_OFFSET =192
num_features = 64
num_hidden = 64
num_layers=1
image_shape=(128,128)
IMAGE_SHAPE_RESULT = (128, 128, 1)
num_epochs=1002
num_batches_per_epoch=100
save_step=num_epochs/3


#onehot representation of labels
def onehot3(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[row, col] = 1
    return out

#read data from csv files
def read_data(train=True):
    if train:
        df = read_csv(os.path.expanduser(TRAIN_DATA_FILE))  # load pandas dataframe
        labels  = df["species"].values #np.vstack
        features = df[df.columns[2:]].values 
    else:
        df = read_csv(os.path.expanduser(TEST_DATA_FILE))  # load pandas dataframe
        features = df[df.columns[1:]].values 
    imageid = df["id"].values
    imageid = imageid.astype(str)
    #imageid = imageid.astype(np.int32)
    #imageid=np.array(map(str, imageid))
    #indices=range(len(imageid))
    #indices = numpy.arange(len(imageid))
    #np.random.shuffle(indices)
    #imageid= imageid[indices]
    #features= features[indices]
    #if train:
    #    labels= labels[indices]
    
    features= features.astype(np.float32)
    features= np.asarray(features[np.newaxis, :])
    #features = tf.transpose(features, [1, 0, 2]).eval()#image = np.expand_dims(image, axis=2)
    #features = np.array(features)
    #imageid = [IMAGE_DIR + row.tostring().decode('UTF-8').strip() + IMAGE_EXT for row in imageid]
    ##loading images from disk
    imageid = [IMAGE_DIR + row + IMAGE_EXT for row in imageid]
    allimage = []
    for image in imageid:
        ds = misc.imread(image)
        ds = misc.imresize(ds, size=image_shape)#np.resize(ds, output_shape=image_shape)
        ds = np.expand_dims(ds, axis=2)#axis=1
        ds = np.expand_dims(ds, axis=0)#axis=1
        #ds= ds[np.newaxis,:]
        allimage.append(ds)
    allimage= np.array(allimage)
    
    if train:
        le = LabelEncoder().fit(labels) 
        labels = le.transform(labels)           # encode species strings
        classes = list(le.classes_)
        labels= np.array(labels)
        labels=onehot3(labels, len(classes))
        return allimage, features, labels, classes
    else :
        return allimage, features

#get training batch; actually batch_size is 1
def next_training_batch(train_image, train_shape, train_margin, train_texture, train_labels, batch):
    import random
    # random_index = random.choice(list())
    #random_index = random.choice(list()[0:5])
    num_examples=  len(train_image)
    random_index = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]
    #training_element = train_image[random_index]
    train_img_batch = train_image[batch % num_examples,:,:]
    train_img_batch = np.asarray(train_img_batch[np.newaxis, :])
    train_shape_batch = train_shape[batch % num_examples,:,:]
    train_shape_batch = np.asarray(train_shape_batch[np.newaxis, :])
    train_texture_batch = train_texture[batch % num_examples,:,:]
    train_texture_batch = np.asarray(train_texture_batch[np.newaxis, :])
    train_margin_batch = train_margin[batch % num_examples,:,:]
    train_margin_batch = np.asarray(train_margin_batch[np.newaxis, :])
    train_label_batch = train_labels[batch % num_examples,:]
    train_label_batch = np.asarray(train_label_batch[np.newaxis, :])

    return train_img_batch, train_shape_batch, train_margin_batch, train_texture_batch, train_label_batch

#validation data batch
def next_validation_batch(valid_image, valid_shape, valid_margin, valid_texture, valid_labels, batch ):
    import random
    # random_index = random.choice(list())
    #random_index = random.choice(list()[0:5])
    num_examples=  len(valid_image)
    random_index = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]
    #training_element = valid_image[random_index]
    test_img_batch = valid_image[batch % num_examples,:,:]
    test_img_batch = np.asarray(test_img_batch[np.newaxis, :])
    test_shape_batch = valid_shape[batch % num_examples,:,:]
    test_shape_batch = np.asarray(test_shape_batch[np.newaxis, :])
    test_texture_batch = valid_texture[batch % num_examples,:,:]
    test_texture_batch = np.asarray(test_texture_batch[np.newaxis, :])
    test_margin_batch = valid_margin[batch % num_examples,:,:]
    test_margin_batch = np.asarray(test_margin_batch[np.newaxis, :])
    test_label_batch = valid_labels[batch % num_examples,:]
    test_label_batch = np.asarray(test_label_batch[np.newaxis, :])
    return test_img_batch, test_shape_batch , test_margin_batch, test_texture_batch, test_label_batch

#test data batch
def next_test_batch(valid_image, valid_shape, valid_margin, valid_texture, batch ):
    import random
    # random_index = random.choice(list())
    #random_index = random.choice(list()[0:5])
    num_examples=  len(valid_image)
    random_index = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]
    #training_element = valid_image[random_index]
    test_img_batch = valid_image[batch % num_examples,:,:]
    test_img_batch = np.asarray(test_img_batch[np.newaxis, :])
    test_shape_batch = valid_shape[batch % num_examples,:,:]
    test_shape_batch = np.asarray(test_shape_batch[np.newaxis, :])
    test_texture_batch = valid_texture[batch % num_examples,:,:]
    test_texture_batch = np.asarray(test_texture_batch[np.newaxis, :])
    test_margin_batch = valid_margin[batch % num_examples,:,:]
    test_margin_batch = np.asarray(test_margin_batch[np.newaxis, :])
    return test_img_batch, test_shape_batch , test_margin_batch, test_texture_batch

#main
if __name__ == "__main__":


    # THE MAIN CODE!
    #Read training data
    #trainimage, trainshape, trainmargin, traintexture,  trainlabels, classes=read_data(True)
    trainimage, trainfeatures,  trainlabels, classes=read_data(True)
    num_classes= len(classes)

    #build graph
    graph = tf.Graph()
    with graph.as_default():
        inputs_shape = tf.placeholder(tf.float32, [None, None, num_features])
        inputs_margin = tf.placeholder(tf.float32, [None, None, num_features])
        inputs_texture = tf.placeholder(tf.float32, [None, None, num_features])
        targets = tf.placeholder(tf.float32, shape=(batch_size, num_classes))  #tf.sparse_placeholder(tf.float32)
        seq_len = tf.placeholder(tf.int32, [None])

        #inputs_shape = tf.split(axis=0, num_or_size_splits=max_input_length, value=inputs_shape)  # n_steps * (batch_size, features)
        #inputs_margin = tf.split(axis=0, num_or_size_splits=max_input_length, value=inputs_margin)  # n_steps * (batch_size, features)
        #inputs_texture = tf.split(axis=0, num_or_size_splits=max_input_length, value=inputs_texture)  # n_steps * (batch_size, features)
        #y = tf.placeholder(tf.float32, shape=(batch_size, num_classes))  # -> seq2seq!
        
        cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
        outputs_shape, _ = tf.nn.dynamic_rnn(stack, inputs_shape, seq_len, dtype=tf.float32)
        outputs_margin, _ = tf.nn.dynamic_rnn(stack, inputs_margin, seq_len, dtype=tf.float32)
        outputs_texture, _ = tf.nn.dynamic_rnn(stack, inputs_texture, seq_len, dtype=tf.float32)
        outputs = tf.concat( values=[outputs_shape, outputs_margin, outputs_texture], axis=1)
        shape = tf.shape(inputs_shape)
        batch_s, max_timesteps = shape[0], shape[1]
        outputs = tf.reshape(outputs, [-1, num_hidden*3])
        W = tf.Variable(tf.truncated_normal([num_hidden*3, num_classes],stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[num_classes]))
        
        prediction =  tf.matmul(outputs, W) + b#tf.nn.softmax
        #tf.contrib.layers.summarize_variables()
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=targets))  # prediction, target
        #optimizer = tf.train.AdamOptimizer(mlearning_rate).minimize(cost)
        optimizer = tf.train.AdamOptimizer()#tf.train.MomentumOptimizer(learning_rate=mlearning_rate, momentum=0.95).minimize(cost)
        minimize = optimizer.minimize(cross_entropy)
        eqeval = tf.equal(tf.argmax(prediction, 1), tf.argmax(targets, 1))
        accuracy = tf.reduce_mean(tf.cast(eqeval, tf.float32))
        mistakes = tf.not_equal(tf.argmax(targets, 1), tf.argmax(prediction, 1))
        error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
        # add TensorBoard summaries for all variables
        #tf.scalar_summary('train/cost', cost)
        #tf.scalar_summary('train/accuracy', accuracy)

    with tf.Session(graph=graph) as session:
        #  format tran data
        #trainfeatures = tf.transpose(trainfeatures, [1, 0, 2]).eval()#image = np.expand_dims(image, axis=2)
        trainfeatures = np.array(trainfeatures)
        #print("trainfeatures:{} ".format(trainfeatures.shape))
        trainmargin= trainfeatures[:,:,:MARGIN_OFFSET]
        trainshape= trainfeatures[:,:,MARGIN_OFFSET:SHAPE_OFFSET]
        traintexture= trainfeatures[:,:,SHAPE_OFFSET:TEXTURE_OFFSET]
        trainmargin = tf.transpose(trainmargin, [1, 0, 2]).eval()#image = np.expand_dims(image, axis=2)
        trainmargin = np.array(trainmargin)
        traintexture = tf.transpose(traintexture, [1, 0, 2]).eval()#image = np.expand_dims(image, axis=2)
        traintexture = np.array(traintexture)
        trainshape = tf.transpose(trainshape, [1, 0, 2]).eval()#image = np.expand_dims(image, axis=2)
        trainshape = np.array(trainshape)
        
        #getting savepoint if exists
        try:saver = tf.train.Saver(tf.global_variables())
        except:saver = tf.train.Saver(tf.global_variables())
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir="checkpoints")
        if checkpoint:
            print("LOADING " + checkpoint + " !!!\n")
            try:saver.restore(session, checkpoint)
            except: print("incompatible checkpoint")

        tf.global_variables_initializer().run()
        #validationimage, validationshape, validationtexture, validationmargin, validationlabels=read_data(True)
    
        for curr_epoch in range(num_epochs):
            train_cost = train_ler = 0
            start = time.time()
    
            for batch in range(num_batches_per_epoch):
        
                #  feed SparseTensor input
                #batch_train_targets = sparse_tuple_from(train_targets[indexes]) 
                batch_train_seq_len=[1]*batch_size
                batchtrainimage, batchtrainshape,  batchtrainmargin, batchtraintexture, batchtrainlabels= next_training_batch(trainimage, trainshape,  trainmargin, traintexture, trainlabels, batch)
                feed = {inputs_shape: batchtrainshape, inputs_margin: batchtrainmargin, inputs_texture: batchtraintexture,
                    targets: batchtrainlabels,
                    seq_len: batch_train_seq_len}

                batch_cost, _ = session.run([cross_entropy, minimize], feed)#optimizer
                train_cost += batch_cost*batch_size
                train_ler += session.run(error, feed_dict=feed)*batch_size

    
            # Metrics mean
            train_cost /= num_batches_per_epoch#num_examples
            train_ler /= num_batches_per_epoch#num_examples
    
            log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}\n"
            print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler, time.time() - start))
    

            #  feed SparseTensor input (validation)
            batch_train_seq_len=[1]*batch_size
            batchtestimage, batchtestshape,  batchtestmargin, batchtesttexture, batchtestlabels= next_validation_batch(trainimage, trainshape,  trainmargin, traintexture, trainlabels, batch)

            val_feed = {inputs_shape: batchtestshape, inputs_margin: batchtestmargin, inputs_texture: batchtesttexture,
                targets: batchtestlabels,
                seq_len: batch_train_seq_len
                }

            # Decoding
            test_pred, test_acc, test_cost = session.run([prediction[0],  cross_entropy, error], feed_dict=val_feed)

            dense_decoded = test_pred
            print("pred {} ".format(dense_decoded)+" Labels {} ".format(batchtestlabels[0][0])+ " cross entropy {} ".format(test_acc)+"  error {} ".format(test_cost)+" train cost {}\n ".format(train_cost))
            if curr_epoch % save_step == 0 and curr_epoch > 0:
                snapshot="trainepoch{}".format(curr_epoch)
                print("SAVING snapshot %s" % snapshot)
                saver.save(session, "checkpoints/" + snapshot + ".ckpt", curr_epoch)


        # get test data
        testimage, testfeatures=read_data(False)
        testfeatures = np.array(testfeatures)
        testmargin= testfeatures[:,:,:MARGIN_OFFSET]
        testshape= testfeatures[:,:,MARGIN_OFFSET:SHAPE_OFFSET]
        testtexture= testfeatures[:,:,SHAPE_OFFSET:TEXTURE_OFFSET]
        testmargin = tf.transpose(testmargin, [1, 0, 2]).eval()#image = np.expand_dims(image, axis=2)
        testmargin = np.array(testmargin)
        testtexture = tf.transpose(testtexture, [1, 0, 2]).eval()#image = np.expand_dims(image, axis=2)
        testtexture = np.array(testtexture)
        testshape = tf.transpose(testshape, [1, 0, 2]).eval()#image = np.expand_dims(image, axis=2)
        testshape = np.array(testshape)
    
        testpredvalue=[]
        testlabelsvalue=[]
        testids=[]
        # feed SparseTensor and output predictions        
        num_test_data=len(testimage)
        for curr_epoch in range(num_test_data):
            batchtestimage, batchtestshape,  batchtestmargin, batchtesttexture= next_test_batch(testimage, testshape,  testmargin, testtexture,  curr_epoch)
            batch_train_seq_len=[1]*batch_size
            val_feed = {inputs_shape: batchtestshape, inputs_margin: batchtestmargin, inputs_texture: batchtesttexture,
                #targets: batchtestlabels,
                seq_len: batch_train_seq_len
                }
            test_pred = session.run([prediction], feed_dict=val_feed)
            dense_decoded = test_pred[0] 
            #print("pred {} ".format(dense_decoded)
            argpred=np.argmax(test_pred[0], 1)
            print("{}  result size {}\n".format(curr_epoch, argpred.shape))
            testpredvalue.append(argpred)
            testids.append(curr_epoch)
        ids_test_df = pd.DataFrame(testids, columns=["id"])
        testresult_df= pd.DataFrame(testpredvalue, columns=["prediction"])
        submission = pd.concat([ids_test_df, testresult_df], axis=1)
        submission.to_csv('testresult_mlp.csv', index=False)
        
