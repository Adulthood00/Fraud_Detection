import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import os
from functions import *

#read data
data_identity = pd.read_csv("train_identity.csv")
data_transaction = pd.read_csv("train_transaction.csv")


data = data_transaction.merge(data_identity,how="outer")


my_list = []
i=569877
index = -1
for row in data["isFraud"]:
    index+=1
    if row == 0:
        my_list.append(index)
        i-=1
        if i < 20663:
            break

data = data.drop(my_list)
data.count()

data["isFraud"].value_counts()


# checking missing data
total = data.isnull().sum().sort_values(ascending = False)
percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


for col in data.columns:
    percent  = (data[col].isnull().sum()/data[col].isnull().count()*100)
    if percent > 70.0:
        data = data.drop(col,axis=1)


#for col in data.columns:
#    print("Number of unique values of {} : {}".format(col, data[col].nunique()))


#data["isFraud"].value_counts().plot.bar()


data = deal_with_missing_values(data)
#check_to_mikos_apo_ta_columns(data)
#check_for_missing_values(data)
#data.count()

data = label_encoder(data)

data.dtypes.unique()

#corr_matrix = data.corr()
#print(corr_matrix["isFraud"])

predictors = data.drop("isFraud", axis=1)
target = data["isFraud"].copy()

predictors = standard_scaler(predictors)

X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.001)

y_train = y_train.values
X_valid = X_test
y_valid = y_test.values

m, n = X_train.shape

n_hidden1 = 1300
n_hidden2 = 1100
n_hidden3 = 500
n_hidden4 = 150
n_outputs = 2

X = tf.placeholder(tf.float32, shape=(None, n), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


with tf.name_scope("fraud_dnn"):
    reg = tf.contrib.layers.l1_l2_regularizer(0.00, 0.00)
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, kernel_regularizer=reg)
    dropout1 = tf.layers.dropout(hidden1, rate=0.0)
    hidden2 = tf.layers.dense(dropout1, n_hidden2, activation=tf.nn.relu, kernel_regularizer=reg)
    dropout2 = tf.layers.dropout(hidden2, rate=0.0)
    hidden3 = tf.layers.dense(dropout2, n_hidden3, activation=tf.nn.relu, kernel_regularizer=reg)
    dropout3 = tf.layers.dropout(hidden3, rate=0.0)
    hidden4 = tf.layers.dense(dropout3, n_hidden4, activation=tf.nn.relu, kernel_regularizer=reg)
    dropout4 = tf.layers.dropout(hidden4, rate=0.0)
    logits = tf.layers.dense(dropout4, n_outputs, name="logits")
    y_proba = tf.nn.softmax(logits, name="y_proba")

with tf.name_scope("fraud_loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_proba)
    loss = tf.reduce_mean(xentropy)
    loss_summary = tf.summary.scalar('log_loss', loss)

learning_rate = 0.01

with tf.name_scope("fraud_train"):
    # optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(y_proba, y, 1, name="correct")
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

from datetime import datetime


def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)


logdir = log_dir("fraud_dnn")
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epochs = 10000
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

checkpoint_path = "/tmp/my_deep_fraud_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "./my_deep_fraud_model"

best_loss = np.infty
epochs_without_progress = 0
max_epochs_without_progress = 500

with tf.Session() as sess:
    if os.path.isfile(checkpoint_epoch_path):
        # if the checkpoint file exists, restore the model and load the epoch number
        with open(checkpoint_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        print("Training was interrupted. Continuing at epoch", start_epoch)
        saver.restore(sess, checkpoint_path)
    else:
        start_epoch = 0
        sess.run(init)
    #start_epoch = 0
    #sess.run(init)
    for epoch in range(start_epoch, n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val, loss_val, accuracy_summary_str, loss_summary_str = sess.run(
            [accuracy, loss, accuracy_summary, loss_summary], feed_dict={X: X_valid, y: y_valid})
        file_writer.add_summary(accuracy_summary_str, epoch)
        file_writer.add_summary(loss_summary_str, epoch)
        if epoch % 100 == 0:
            print("Epoch:", epoch,
                  "\tValidation accuracy: {:.3f}%".format(accuracy_val * 100),
                  "\tLoss: {:.5f}".format(loss_val))
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, "wb") as f:
                f.write(b"%d" % (epoch + 1))
            if loss_val < best_loss:
                saver.save(sess, final_model_path)
                best_loss = loss_val
            else:
                epochs_without_progress += 100
                if epochs_without_progress > max_epochs_without_progress:
                    print("Early stopping")
                    break
