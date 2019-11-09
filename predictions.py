import pandas as pd
import tensorflow as tf
from functions import *

test_transaction = pd.read_csv("test_transaction.csv")
test_identity = pd.read_csv("test_identity.csv")

data = test_transaction.merge(test_identity,how="outer")

data = deal_with_missing_values(data)
check_to_mikos_apo_ta_columns(data)
check_for_missing_values(data)
data.count()

data = label_encoder(data)

data.dtypes.unique()

predictors = data

predictors = standard_scaler(predictors)

with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('my_deep_fraud_model.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./'))

graph=tf.get_default_graph()
graph.get_operations()[100:200]



X=graph.get_tensor_by_name("X:0")

#accuracy=graph.get_tensor_by_name("Cast:0")
#correct=graph.get_tensor_by_name("in_top_k/InTopKV2/k:0")
softmax=graph.get_tensor_by_name("fraud_dnn/y_proba:0")
#logits=graph.get_tensor_by_name("dense_3/kernel/Initializer/random_uniform/shape:0")
feed_dict = {X:data}


with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('my_deep_fraud_model.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./'))
  #predictions=sess.run(correct,feed_dict)
  predictions=sess.run(softmax,feed_dict)
  #predictions=correct.eval(feed_dict)

  
finals=[]
for i in range(len(predictions)):
    if predictions[i][0]>predictions[i][1]:
        finals.append(0)
    else:
        finals.append(1)
        
    


ID=[]
for i in ID_code:
    ID.append(i)


with open("submission.txt","w") as file:
    for i in range(len(ID)):
        wr="{},{}\n".format(ID[i],finals[i])
        file.write(wr)



