import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

df = pd.read_csv('data/associations.csv')
df.rename(columns={'0': 'thm', '1': 'neighbour_thm'}, inplace=True)

vocab = pd.concat([df['thm'], df['neighbour_thm']]).unique()
encoding = pd.Series(vocab).to_frame()
enc = encoding.reset_index().rename(columns={'index': 'label'})
enc = enc.set_index(0)['label']
df['thm_label'] = df['thm'].map(enc)
df['neighbour_thm_label'] = df['neighbour_thm'].map(enc)

# definitions
EMBEDDING_SIZE = 500
VOCAB_SIZE = vocab.shape[0]

NEGATIVE_SAMPLES = 3

BATCH_SIZE = 32
EVAL_BATCH_SIZE = 200000

embeddings = tf.Variable(
    tf.random_uniform([VOCAB_SIZE, EMBEDDING_SIZE], -1.0, 1.0))
import math

nce_weights = tf.Variable(
    tf.truncated_normal([VOCAB_SIZE, EMBEDDING_SIZE],
                        stddev=1.0 / math.sqrt(EMBEDDING_SIZE)))
nce_biases = tf.Variable(tf.zeros([VOCAB_SIZE]))
train_inputs = tf.placeholder(tf.int32, shape=[None])
train_labels = tf.placeholder(tf.int32, shape=[None, 1])

embed = tf.nn.embedding_lookup(embeddings, train_inputs)
loss = tf.reduce_mean(
    tf.nn.nce_loss(weights=nce_weights,
                   biases=nce_biases,
                   labels=train_labels,
                   inputs=embed,
                   num_sampled=NEGATIVE_SAMPLES,
                   num_classes=VOCAB_SIZE))

optimizer = tf.train.AdamOptimizer(learning_rate=0.000025).minimize(loss)


# training


def whole_set(method='skip_gram'):
    if method == 'skip_gram':
        inputs = df['thm_label'].values
        labels = df['neighbour_thm_label'].values.reshape(-1, 1)
    elif method == 'CBOW':
        inputs = df['neighbour_thm_label'].values
        labels = df['thm_label'].values.reshape(-1, 1)
    else:
        raise Exception('Wrong training method')
    return inputs, labels


all_inputs, all_labels = whole_set()


def generate_batch():
    """
    Generates minibatch of size BATCH_SIZE for training accordingly to the method ('skip_gram' or 'CBOW').
    Returns:
        inputs - vector of inputs for training
        labels - vector of labels for training
    """

    for epoch_num in range(10000):
        for batch_lower_bound in range(0, df.shape[0], BATCH_SIZE):
            batch_upper_bound = batch_lower_bound + BATCH_SIZE
            yield (all_inputs[batch_lower_bound: batch_upper_bound], all_labels[batch_lower_bound:batch_upper_bound])


# Summaries
with tf.name_scope('Performance'):
    BATCH_LOSS_PH = tf.placeholder(tf.float32, shape=None, name='loss_summary')
    BATCH_LOSS_SUMMARY = tf.summary.scalar('loss', BATCH_LOSS_PH)
    LOSS_PH = tf.placeholder(tf.float32, shape=None, name='loss_summary')
    LOSS_SUMMARY = tf.summary.scalar('loss', LOSS_PH)
PERFORMANCE_SUMMARIES = tf.summary.merge([BATCH_LOSS_SUMMARY, LOSS_SUMMARY])

SUMMARIES = 'summaries'
RUNID = f'run_{1}'
SUMM_WRITER = tf.summary.FileWriter(os.path.join(SUMMARIES, RUNID))

saver = tf.train.Saver()

conf = tf.ConfigProto(log_device_placement=True)

conf.gpu_options.allow_growth = True
with tf.Session(config=conf) as sess:
    sess.run(tf.global_variables_initializer())
    for i, (inputs, labels) in enumerate(generate_batch()):
        feed_dict = {train_inputs: inputs, train_labels: labels}
        _, cur_loss = sess.run([optimizer, loss], feed_dict=feed_dict)
        summ_loss = sess.run(BATCH_LOSS_SUMMARY, feed_dict={BATCH_LOSS_PH: cur_loss})
        SUMM_WRITER.add_summary(summ_loss, i)
        if i % 2000 == 0:
            inds = np.random.choice(all_inputs.shape[0], EVAL_BATCH_SIZE)
            feed_dict = {train_inputs: all_inputs[inds], train_labels: all_labels[inds]}
            cur_loss = sess.run(loss, feed_dict=feed_dict)
            tf.summary.scalar('loss', cur_loss)

            summ_loss = sess.run(LOSS_SUMMARY, feed_dict={LOSS_PH:cur_loss})
            SUMM_WRITER.add_summary(summ_loss, i)

            print(i, cur_loss)
            if i % 100000 == 0 and i > 0:
                save_path = saver.save(sess, "models/model3.ckpt")
                embs = sess.run(embeddings)
                df_embs = pd.DataFrame(embs)
                df_embs['prem'] = vocab
                df_embs.to_csv('data/embeddings_live.csv')
