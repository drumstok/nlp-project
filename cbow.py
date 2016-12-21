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
init_embeddings_pickle_path = 'init_embeddings/'
first_results_path = 'first_training_results/'
# Number of steps after which train loss is printed and vars saved
print_freq = 250
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


def train(seqs_train, seqs_test,
          initial_embeddings=None, initial_weights=None):
    """Train."""
    with graph.as_default(), tf.Session() as sess:
        sess.run(init)

        if initial_embeddings:
            sess.run(embeddings.assign(initial_embeddings))
        if initial_weights:
            sess.run(weights.assign(initial_weights))

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


def adapt_random(initial_embeddings, initial_weights, oov_widxs):
    """Adapt embeddings randomly."""
    # For embeddings, random init with equals norm as rest
    oov_embeddings = np.random.normal(0, 1, [oov_widxs.size, embedding_size])
    embeddings_avg_norm = np.mean(np.linalg.norm(initial_embeddings, axis=1))
    oov_embeddings = oov_embeddings \
        / np.linalg.norm(oov_embeddings, axis=1)[:, np.newaxis] \
        * embeddings_avg_norm
    adapted_embeddings = initial_embeddings.copy()
    adapted_embeddings[oov_widxs, :] = oov_embeddings

    # For weights
    oov_weights = np.random.normal(0, 1, [oov_widxs.size, embedding_size])
    weights_avg_norm = np.mean(np.linalg.norm(initial_weights, axis=1))
    oov_weights = oov_weights \
        / np.linalg.norm(oov_weights, axis=1)[:, np.newaxis] \
        * weights_avg_norm
    adapted_weights = initial_weights.copy()
    adapted_weights[oov_widxs, :] = oov_weights

    return adapted_embeddings, adapted_weights


def adapt_discr(initial_embeddings, initial_weights, oov_widxs, sim_widxs):
    """Adapt embeddings discriminatively."""
    adapted_embeddings = initial_embeddings.copy()
    adapted_weights = initial_weights.copy()

    adapted_embeddings[oov_widxs, :] = adapted_embeddings[sim_widxs, :]
    adapted_weights[oov_widxs, :] = adapted_weights[sim_widxs, :]

    return adapted_embeddings, adapted_weights


def adapt_prob(initial_embeddings, initial_weights,
               oov_widxs, oov_seqs_train):
    """Adapt model probabilistically."""
    # First random init for oov co-occurrence.
    oov_random_embeddings, oov_random_weights = adapt_random(
        initial_embeddings, initial_weights, oov_widxs)

    # Find relevant contexts
    contexts, labels = BatchCreator(oov_seqs_train).create(
        oov_seqs_train.shape[0]*contexts_per_sequence)
    mask = np.in1d(labels, oov_widxs)
    contexts, labels = contexts[mask, :], labels[mask]

    classifications_sum = np.zeros([vocabulary_size, vocabulary_size])
    label_count = np.zeros([vocabulary_size])

    # Compute in batches, then average
    n_steps = np.ceil(labels.size/batch_size).astype(np.int32)
    for step in range(n_steps):
        print("Batch {} / {}".format(step+1, n_steps))
        batch_contexts = contexts[step*batch_size:(step+1)*batch_size]
        batch_labels = labels[step*batch_size:(step+1)*batch_size]
        embedded = oov_random_embeddings[batch_contexts, :]
        # embedded = np.einsum('nmw,wi->nmi',
        #                      batch_contexts, oov_random_embeddings)
        context_vectors = np.mean(embedded, axis=1)
        logits = np.matmul(context_vectors, oov_random_weights.T)
        exponents = np.exp(logits)
        # Don't classify OOV as OOV
        exponents[:, oov_widxs] = 0
        normalizer = np.sum(exponents, axis=1)[:, np.newaxis]
        probabilities = exponents / normalizer
        for prob, label in zip(probabilities, batch_labels):
            classifications_sum[label] += prob
            label_count[label] += 1

    # n_oov x vocab_size matrix indicating final classifications
    classifications = classifications_sum[oov_widxs, :] / \
        label_count[oov_widxs, np.newaxis]
    oov_embeddings = np.matmul(classifications, oov_random_embeddings)
    oov_weights = np.matmul(classifications, oov_random_weights)
    adapted_embeddings = initial_embeddings.copy()
    adapted_embeddings[oov_widxs, :] = oov_embeddings
    adapted_weights = initial_weights.copy()
    adapted_weights[oov_widxs, :] = oov_weights

    return adapted_embeddings, adapted_weights


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

    elif FLAGS.job == 'adapt-random-init':
        initial_embeddings, initial_weights = load_embeddings(
            first_results_path+"/saved/initial")
        adapted_embeddings, adapted_weights = \
            adapt_random(initial_embeddings, initial_weights, oov_widxs)
        joblib.dump((adapted_embeddings, adapted_weights),
                    init_embeddings_pickle_path+"random")
        print("Random init saved")
    elif FLAGS.job == 'adapt-prob-init':
        initial_embeddings, initial_weights = load_embeddings(
            first_results_path+"/saved/initial")
        adapted_embeddings, adapted_weights = \
            adapt_prob(initial_embeddings, initial_weights,
                       oov_widxs, seqs[oov_seqs_train_idxs])
        joblib.dump((adapted_embeddings, adapted_weights),
                    init_embeddings_pickle_path+"prob")
        print("Probabilistic init saved")
    elif FLAGS.job == 'adapt-random-train':
        adapted_embeddings, adapted_weights = joblib.load(
            init_embeddings_pickle_path+"random")
        train(seqs[oov_seqs_train_idxs], seqs[oov_seqs_test_idxs],
              adapted_embeddings, adapted_weights)

    elif FLAGS.job == 'load':
        embeddings_val, weights_val = load_embeddings(FLAGS.saveddir)
        print(embeddings_val.shape, embeddings_val.std(),
              weights_val.shape, weights_val.std())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=str,
                        help='Either of [initial, baseline, adapt-random,'
                             ' adapt-descr, adapt-prob]')
    parser.add_argument('--logdir', type=str, default='default',
                        help='Relative log and checkpoint path')
    parser.add_argument('--saveddir', type=str,
                        help='Path to found embeddings')
    parser.add_argument('--save_adapted', type=str,
                        help='Path where to store adapted embeddings')
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
