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
                                stddev=1.0/math.sqrt(embedding_size)),
            name="embeddings")

    embedded = tf.nn.embedding_lookup(embeddings, inputs_placeholder)

    weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0/math.sqrt(embedding_size)),
            name="weights")
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


def train(seqs_train, seqs_test, initialize_op=None):
    """Train."""
    with graph.as_default(), tf.Session() as sess:
        sess.run(init)

        if initialize_op:
            sess.run(initialize_op)

        saver = tf.train.Saver()
        writer = tf.train.SummaryWriter(
            log_dir+FLAGS.logdir+"/"+FLAGS.job, graph=sess.graph)
        train_batch_creator = BatchCreator(seqs_train)
        print("Initialized")

        average_loss = 0
        average_time = 0
        for step in range(num_steps):
            start_time = time.time()
            batch_inputs, batch_labels = train_batch_creator.create(batch_size)
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
                saver.save(sess, save_dir+FLAGS.logdir+"/"+FLAGS.job+"/saved",
                           global_step=global_step_val)

            if step % eval_freq == 0 or step == num_steps-1:
                for name, seqs_eval in [['train', seqs_train],
                                        ['test', seqs_test]]:
                    eval_batch_creator = BatchCreator(seqs_eval)
                    start_time = time.time()
                    cum_loss = 0.
                    for _ in range(eval_steps):
                        eval_inputs, eval_labels = eval_batch_creator.create(
                            eval_batch_size)
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


class BatchCreator(object):
    """Batch creator class. Shuffles once."""

    def __init__(self, seqs):
        """Initialise."""
        self.seqs = np.copy(seqs)
        self.size = seqs.shape[0]
        np.random.shuffle(self.seqs)
        self.index = 0

    def create(self, batch_size):
        """Create batch."""
        n_seqs = np.ceil(batch_size/contexts_per_sequence).astype(np.int32)
        if self.index + n_seqs >= self.size-1:
            np.random.shuffle(self.seqs)
            self.index = 0

        batch_seqs = self.seqs[self.index:self.index+n_seqs]
        self.index += n_seqs
        return cpl.sample_cbow_batch(batch_seqs, cbow_context_size)


def load_embeddings(path):
    """Load embeddings from path to NumPy array."""
    with graph.as_default(), tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(path))
        return sess.run([embeddings, weights])


def adopt_random(initial_embeddings, initial_weights, oov_widxs):
    """Adopt embeddings randomly."""
    r1 = np.random.normal(0, 1.0/math.sqrt(embedding_size),
                          [oov_widxs.size, embedding_size])
    r2 = np.random.normal(0, 1.0/math.sqrt(embedding_size),
                          [oov_widxs.size, embedding_size])
    adopted_embeddings = initial_embeddings.copy()
    adopted_embeddings[oov_widxs, :] = r1
    adopted_weights = initial_weights.copy()
    adopted_weights[oov_widxs, :] = r2

    with graph.as_default():
        op = [embeddings.addign(adopted_embeddings),
              weights.assign(adopted_weights)]
        return op


def main(_):
    """Execute main function."""
    data = cpl.load_data(data_path=data_path)
    zipf_idxs, inv_zipf_idx, counts = cpl.create_zipf_index(data)
    data = cpl.zipf(data, zipf_idxs)
    data = cpl.unknownize(data, vocabulary_size)
    seqs = cpl.to_sequences(data)

    if FLAGS.job == 'sample-oov':
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
        return
    else:
        if not os.path.exists(sample_pickle_path):
            print("No OOV sample found")
            return
        packet = joblib.load(sample_pickle_path)
        oov_widxs, (oov_seqs_idxs, iv_seqs_idxs), \
            (oov_seqs_train_idxs, oov_seqs_test_idxs,
             iv_seqs_train_idxs, iv_seqs_test_idxs) = packet
        print("Sample loaded")

    if FLAGS.job == 'initial':
        print("Running initial job")
        train(seqs[iv_seqs_train_idxs], seqs[iv_seqs_test_idxs])

    elif FLAGS.job == 'baseline':
        print("Running baseline job")
        train_idx = np.concatenate((oov_seqs_train_idxs, iv_seqs_train_idxs))
        test_idx = np.concatenate((oov_seqs_test_idxs, iv_seqs_test_idxs))
        train(seqs[train_idx], seqs[test_idx])

    elif FLAGS.job == 'adopt-random':
        if not FLAGS.saveddir:
            print("No saveddir given")
            return
        print(oov_widxs.max())
        return
        embeddings_val, weights_val = load_embeddings(FLAGS.saveddir)
        init_op = adopt_random(embeddings_val, weights_val, oov_widxs)
        train(seqs[oov_seqs_train_idxs], seqs[oov_seqs_test_idxs], init_op)

    elif FLAGS.job == 'load':
        if not FLAGS.saveddir:
            print("No saveddir given")
            return
        embeddings_val, weights_val = load_embeddings(FLAGS.saveddir)
        print(embeddings_val.shape, embeddings_val.std(),
              weights_val.shape, weights_val.std())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=str,
                        help='Either of [initial, baseline]')
    parser.add_argument('--logdir', type=str, default='default',
                        help='Relative log and checkpoint path')
    parser.add_argument('--saveddir', type=str,
                        help='Path to found embeddings')
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
