"""OOV CBOW model."""
import tensorflow as tf
import numpy as np
import argparse
import math
import corpuslib as cpl
import time
from sklearn.externals import joblib
import os

vocabulary_size = 30000
num_oov_words = 1000
oov_cutoff_left = 1000

batch_size = 1024
embedding_size = 300  # Dimension of the embedding vector.
cbow_context_size = 4  # How many times to reuse an input to generate a label.
contexts_per_sequence = 16

num_softmax_sampled = 2000  # Number of negative examples to sample in softmax
num_steps = 40000000
save_dir = 'saved/'
log_dir = 'logs/'
data_path = 'parsed/indices.npy'
sample_pickle_path = 'sample_pickle/sample_pickle.pkl'
# Number of steps after which train loss is printed and vars saved
print_freq = 100
# Number of steps after which full loss is evaluated and summaries saved
eval_freq = 1000
# Number of samples used for evaluation
eval_steps = 20
eval_batch_size = 30000

graph = tf.Graph()

with graph.as_default():
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Input data.
    inputs_placeholder = tf.placeholder(
        tf.int32, shape=[None, cbow_context_size])
    labels_placeholder = tf.placeholder(tf.int32, shape=[None])

    embeddings = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0/math.sqrt(embedding_size)))

    embedded = tf.nn.embedding_lookup(embeddings, inputs_placeholder)

    weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0/math.sqrt(embedding_size)))
    biases = tf.zeros([vocabulary_size])

    context_vectors = tf.reduce_mean(embedded, 1)

    # sampled_losses = tf.nn.sampled_softmax_loss(
    #     weights, biases, context_vectors,
    #     tf.reshape(labels_placeholder,
    #                [tf.size(labels_placeholder), 1]),
    #     num_softmax_sampled,
    #     vocabulary_size)
    # loss = tf.reduce_mean(sampled_losses)

    logits = tf.matmul(context_vectors, tf.transpose(weights))
    full_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels_placeholder))

    loss = full_loss

    # train_loss_summary = tf.scalar_summary("train-loss", loss)
    # train_full_loss_summary = tf.scalar_summary("train-full-loss", full_loss)
    # test_loss_summary = tf.scalar_summary("test-loss", loss)
    # test_full_loss_summary = tf.scalar_summary("test-full-loss", full_loss)

    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = optimizer.minimize(loss, global_step=global_step)

    init = tf.initialize_all_variables()


def train(seqs_train, seqs_test):
    """Train."""
    with tf.Session(graph=graph) as sess:
        sess.run(init)
        saver = tf.train.Saver()
        writer = tf.train.SummaryWriter(log_dir+FLAGS.job, graph=graph)
        print("Initialized")

        average_loss = 0
        average_time = 0
        for step in range(num_steps):
            start_time = time.time()
            batch_inputs, batch_labels = create_batch(seqs_train, batch_size)
            feed_dict = {
                inputs_placeholder: batch_inputs,
                labels_placeholder: batch_labels}

            _, loss_val, global_step_val = sess.run(
                [train_op, loss, global_step], feed_dict=feed_dict)
            average_loss += loss_val
            average_time += time.time() - start_time

            if step % print_freq == 0 or step == num_steps-1:
                if step > 0:
                    average_loss /= print_freq
                    average_time /= print_freq
                template = "Average loss at step {}: {:.3f} ({:.1f}ms)"
                print(template.format(
                    global_step_val, average_loss, average_time*1000))
                average_loss = 0
                average_time = 0
                saver.save(sess, save_dir+FLAGS.job+"/saved",
                           global_step=global_step_val)

            if step % eval_freq == 0 or step == num_steps-1:
                for name, seqs_eval in [['train', seqs_train],
                                        ['test', seqs_test]]:
                    start_time = time.time()
                    cum_loss = 0.
                    for _ in range(eval_steps):
                        eval_inputs, eval_labels = create_batch(
                            seqs_eval, eval_batch_size)
                        eval_feed_dict = {
                            inputs_placeholder: eval_inputs,
                            labels_placeholder: eval_labels}
                        cum_loss += sess.run(
                            full_loss, feed_dict=eval_feed_dict)
                    eval_loss = cum_loss / eval_steps
                    duration = time.time() - start_time
                    summary = tf.Summary(value=[
                        tf.Summary.Value(tag="{}-loss".format(name),
                                         simple_value=eval_loss),
                    ])
                    writer.add_summary(summary, global_step_val)
                    template = "Evaluated {} loss at step {}: {:.3f} ({:.1f}s)"
                    print(template.format(
                        name, global_step_val, eval_loss, duration))
                writer.flush()


def create_batch(seqs, batch_size):
    """Create from a sequence batch_size CBOW contexts and labels."""
    sample_idxs = cpl.sample_seqs(seqs, np.ceil(
        batch_size/contexts_per_sequence).astype(np.int32))

    contexts, targets = cpl.sample_cbow_batch(
        seqs[sample_idxs], cbow_context_size)
    return contexts[:batch_size], targets[:batch_size]


def main(_):
    """Execute main function."""
    data = cpl.load_data(data_path=data_path)
    zipf_idxs, inv_zipf_idx, counts = cpl.create_zipf_index(data)
    data = cpl.zipf(data, zipf_idxs)
    data = cpl.unknownize(data, vocabulary_size)
    seqs = cpl.to_sequences(data)

    if not os.path.exists(sample_pickle_path):
        oov_widxs, (oov_seqs_idxs, iv_seqs_idxs) = cpl.sample_oov_seqs(
            seqs, zipf_idxs, num_oov_words, oov_cutoff_left, vocabulary_size-1)
        oov_seqs_train_idxs, oov_seqs_test_idxs, \
            iv_seqs_train_idxs, iv_seqs_test_idxs = \
            cpl.make_train_test_split(oov_seqs_idxs, iv_seqs_idxs)
        packet = oov_widxs, (oov_seqs_idxs, iv_seqs_idxs), \
            (oov_seqs_train_idxs, oov_seqs_test_idxs,
             iv_seqs_train_idxs, iv_seqs_test_idxs)
        joblib.dump(packet, sample_pickle_path)
        print("Sample created")
    else:
        packet = joblib.load(sample_pickle_path)
        oov_widxs, (oov_seqs_idxs, iv_seqs_idxs), \
            (oov_seqs_train_idxs, oov_seqs_test_idxs,
             iv_seqs_train_idxs, iv_seqs_test_idxs) = packet
        print("Sample loaded")

    if FLAGS.job == 'initial':
        print("Running initial job")
        train(seqs[oov_seqs_train_idxs], seqs[oov_seqs_test_idxs])

    elif FLAGS.job == 'baseline':
        print("Running baseline job")
        train_idx = np.concatenate((oov_seqs_train_idxs, iv_seqs_train_idxs))
        test_idx = np.concatenate((oov_seqs_test_idxs, iv_seqs_test_idxs))
        train(seqs[train_idx], seqs[test_idx])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type = str,
                        help='Either of [initial, baseline]')
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
