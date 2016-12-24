"""OOV CBOW model."""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import argparse
import math
import corpuslib as cpl
import time
from sklearn.externals import joblib
from scipy.stats import ttest_ind
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
eval_pickle_path = 'evaluations/'
# Number of steps after which train loss is printed and vars saved
print_freq = 250
# Number of steps after which full loss is evaluated and summaries saved
eval_freq = 500
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

    logits = tf.matmul(context_vectors, tf.transpose(weights))
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels_placeholder)
    loss = tf.reduce_mean(losses)

    probabilities = tf.nn.softmax(logits)

    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = optimizer.minimize(loss, global_step=global_step)

    init = tf.initialize_all_variables()


def train(seqs_train, seqs_test,
          log_dir, save_dir,
          initial_embeddings=None, initial_weights=None):
    """Train."""
    with graph.as_default(), tf.Session() as sess:
        sess.run(init)

        if initial_embeddings is not None:
            sess.run(embeddings.assign(initial_embeddings))
        if initial_weights is not None:
            sess.run(weights.assign(initial_weights))

        saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.2)
        writer = tf.train.SummaryWriter(log_dir, graph=sess.graph)
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
                saver.save(sess, save_dir, global_step=global_step_val)

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
                            loss, feed_dict=eval_feed_dict)
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
        saver.restore(sess, path)
        return sess.run([embeddings, weights])


def adapt_random(initial_embeddings, initial_weights, oov_widxs):
    """Adapt embeddings randomly."""
    # For embeddings, random init with equals norm as rest
    embeddings_avg_norm = np.mean(np.linalg.norm(initial_embeddings, axis=1))
    embeddings_avg_norm = initial_embeddings.std()
    oov_embeddings = np.random.normal(0, embeddings_avg_norm,
                                      [oov_widxs.size, embedding_size])
    adapted_embeddings = initial_embeddings.copy()
    adapted_embeddings[oov_widxs, :] = oov_embeddings

    # For weights
    weights_avg_norm = np.mean(np.linalg.norm(initial_weights, axis=1))
    weights_avg_norm = initial_weights.std()
    oov_weights = np.random.normal(0, weights_avg_norm,
                                   [oov_widxs.size, embedding_size])
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
    """Adapt model probabilistically.

    Input matrices should have random OOV vectors.
    """
    # Find relevant contexts
    contexts, labels = BatchCreator(oov_seqs_train).create(
        oov_seqs_train.shape[0]*contexts_per_sequence)
    mask = np.in1d(labels, oov_widxs)
    contexts, labels = contexts[mask, :], labels[mask]

    classifications_sum = np.zeros([vocabulary_size, vocabulary_size])
    label_count = np.zeros([vocabulary_size])

    # Don't classify OOV as OOV
    weights_oov_zero = initial_weights.copy()
    weights_oov_zero[oov_widxs, :] = 0

    # Compute in batches, then average
    n_steps = np.ceil(labels.size/batch_size).astype(np.int32)
    for step in range(n_steps):
        print("Batch {} / {}".format(step+1, n_steps))
        batch_contexts = contexts[step*batch_size:(step+1)*batch_size]
        batch_labels = labels[step*batch_size:(step+1)*batch_size]
        with graph.as_default(), tf.Session() as sess:
            probs = sess.run(probabilities, feed_dict={
                inputs_placeholder: batch_contexts,
                embeddings: initial_embeddings,
                weights: weights_oov_zero
            })
        counts = np.bincount(batch_labels)
        label_count[np.arange(counts.size)] += counts
        for prob, label in zip(probs, batch_labels):
            classifications_sum[label] += prob

    # n_oov x vocab_size matrix indicating final classifications
    classifications = classifications_sum[oov_widxs, :] / \
        label_count[oov_widxs, np.newaxis]

    oov_embeddings = np.matmul(classifications, initial_embeddings)
    oov_weights = np.matmul(classifications, initial_weights)

    # Remove NaN for numbers that have not been target anywhere
    mask = ~np.isnan(classifications).any(axis=1)

    adapted_embeddings = initial_embeddings.copy()
    adapted_embeddings[oov_widxs[mask], :] = oov_embeddings[mask, :]
    adapted_weights = initial_weights.copy()
    adapted_weights[oov_widxs[mask], :] = oov_weights[mask, :]

    return adapted_embeddings, adapted_weights


def loss_ov_iv(embeddings_val, weights_val, oov_widxs, seqs):
    """Evaluate loss for OOV and IV words separately."""
    batch_creator = BatchCreator(seqs)

    iv_losses = []
    oov_losses = []
    with graph.as_default(), tf.Session() as sess:
        for step in range(eval_steps):
            batch_contexts, batch_labels = batch_creator.create(
                eval_batch_size)
            losses_val = sess.run(losses, feed_dict={
                inputs_placeholder: batch_contexts,
                labels_placeholder: batch_labels,
                embeddings: embeddings_val,
                weights: weights_val
            })
            mask = np.in1d(batch_labels, oov_widxs)
            iv_losses += losses_val[~mask].tolist()
            oov_losses += losses_val[mask].tolist()
    return np.array(iv_losses), np.array(oov_losses)


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
        train(seqs[iv_seqs_train_idxs], seqs[iv_seqs_test_idxs],
              log_dir+'initial', save_dir+'initial')

    elif FLAGS.job == 'baseline':
        print("Running baseline job")
        train_idx = np.concatenate((oov_seqs_train_idxs, iv_seqs_train_idxs))
        test_idx = np.concatenate((oov_seqs_test_idxs, iv_seqs_test_idxs))
        train(seqs[train_idx], seqs[test_idx],
              log_dir+'baseline', save_dir+'baseline')

    elif FLAGS.job == 'adapt-init-random':
        initial_embeddings, initial_weights = joblib.load(
            init_embeddings_pickle_path+"initial")
        adapted_embeddings, adapted_weights = \
            adapt_random(initial_embeddings, initial_weights, oov_widxs)
        joblib.dump((adapted_embeddings, adapted_weights),
                    init_embeddings_pickle_path+"random")
        print("Random init saved")

    elif FLAGS.job == 'adapt-init-prob':
        if not os.path.exists(init_embeddings_pickle_path+"random"):
            print("Ranom init not found")
            return
        random_embeddings, random_weights = joblib.load(
            init_embeddings_pickle_path+"random")
        adapted_embeddings, adapted_weights = adapt_prob(
            random_embeddings, random_weights,
            oov_widxs, seqs[oov_seqs_train_idxs])
        joblib.dump((adapted_embeddings, adapted_weights),
                    init_embeddings_pickle_path+"prob")
        print("Probabilistic init saved")

    elif FLAGS.job == 'adapt-train-random':
        adapted_embeddings, adapted_weights = joblib.load(
            init_embeddings_pickle_path+"random")
        train(seqs[oov_seqs_train_idxs], seqs[oov_seqs_test_idxs],
              log_dir+'adapt-random', save_dir+'adapt-random/saved',
              adapted_embeddings, adapted_weights)

    elif FLAGS.job == 'adapt-train-prob':
        adapted_embeddings, adapted_weights = joblib.load(
            init_embeddings_pickle_path+"prob")
        train(seqs[oov_seqs_train_idxs], seqs[oov_seqs_test_idxs],
              log_dir+'adapt-prob', save_dir+'adapt-prob/saved',
              adapted_embeddings, adapted_weights)

    elif FLAGS.job == 'eval':
        embedding_names = ['baseline', 'initial', 'prob', 'random',
                           'trained-prob', 'trained-random']
        for name in embedding_names:
            embeddings, weights = joblib.load(
                init_embeddings_pickle_path+name)
            iv_losses, oov_losses = loss_ov_iv(
                embeddings, weights, oov_widxs, seqs[oov_seqs_test_idxs])
            joblib.dump((iv_losses, oov_losses), eval_pickle_path+name)
            print("For {}".format(name))
            print("IV losses: mean {:.4f} std {:.4f}".format(
                np.mean(iv_losses), np.std(iv_losses)))
            print("OOV losses: mean {:.4f} std {:.4f}".format(
                np.mean(oov_losses), np.std(oov_losses)))

    elif FLAGS.job == 'stats':
        comparisons = [['random', 'prob'], ['trained-random', 'trained-prob']]
        for x, y in comparisons:
            x_iv_losses, x_oov_losses = joblib.load(eval_pickle_path+x)
            y_iv_losses, y_oov_losses = joblib.load(eval_pickle_path+y)
            iv_stats, iv_p = ttest_ind(x_iv_losses, y_iv_losses)
            oov_stats, oov_p = ttest_ind(x_oov_losses, y_oov_losses)
            print("Comparing {} with {}".format(x, y))
            print("IV t-statistic={} p-value={}".format(
                iv_stats, iv_p))
            print("OOV t-statistic={} p-value={}".format(
                oov_stats, oov_p))

    elif FLAGS.job == 'load':
        embeddings, weights = load_embeddings(
            first_results_path+"/saved/initial/saved-236001")
        joblib.dump((embeddings, weights),
                    init_embeddings_pickle_path+"initial")
        embeddings, weights = load_embeddings(
            first_results_path+"/saved/baseline/saved-233501")
        joblib.dump((embeddings, weights),
                    init_embeddings_pickle_path+"baseline")
        embeddings, weights = load_embeddings(
            "third_training_results/saved/adapt-prob/saved-6501")
        joblib.dump((embeddings, weights),
                    init_embeddings_pickle_path+"trained-prob")
        embeddings, weights = load_embeddings(
            "third_training_results/saved/adapt-random/saved-6501")
        joblib.dump((embeddings, weights),
                    init_embeddings_pickle_path+"trained-random")
        print("Embeddings extracted from TF into NPY.")

    else:
        print("Job not recognised.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=str,
                        help='Either of [initial, baseline, adapt-random,'
                             ' adapt-descr, adapt-prob]')
    parser.add_argument('--logdir', type=str, default='default',
                        help='Relative log and checkpoint path')
    parser.add_argument('--embedding', type=str,
                        help='Embedding to be evaluated')
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
