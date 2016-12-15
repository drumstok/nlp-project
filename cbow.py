import numpy as np
import tensorflow as tf

vocabulary_size = 30000

batch_size = 128
embedding_size = 300    # Dimension of the embedding vector.
cbow_window = 4                 # How many times to reuse an input to generate a label.

num_nce_sampled = 20        # Number of negative examples to sample in NCE
num_steps = 40001
save_dir = 'saved/tryout'
log_dir = 'logs/tryout'

print_freq = 2000
eval_freq = 2000
eval_batch_size = 10000

graph = tf.Graph()

with graph.as_default():
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Input data.
    # train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size, cbow_window])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    embeddings = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))

    embedded = tf.nn.embedding_lookup(embeddings, train_inputs)

    nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]), trainable=False)
    loss = tf.reduce_mean(
            tf.nn.nce_loss(nce_weights, nce_biases, embedded, train_labels,
                           num_nce_sampled, vocabulary_size))

    full_loss = tf.nn.softmax(tf.matmul(embedded, tf.transpose(nce_weights)) + nce_biases)

    train_loss_summary = tf.scalar_summary("train-loss", loss)
    train_full_loss_summary = tf.scalar_summary("train-full-loss", full_loss)
    test_loss_summary = tf.scalar_summary("test-loss", loss)
    test_full_loss_summary = tf.scalar_summary("test-full-loss", full_loss)

    optimizer = tf.train.AdamOptimizer(1e-4, global_loss=global_loss).minimize(loss)

    # Add variable initializer.
    init = tf.initialize_all_variables()

# Step 5: Begin training.
def train():
    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        session.run(init)
        saver = tf.train.Saver({
            'embeddings': embeddings,
            'nce_weights': nce_weights
        })
        writer = tf.train.SummaryWriter(log_dir, graph=graph)
        print("Initialized")

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, cbow_window, skip_window, model)
            feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % print_freq == 0 or step == num_steps-1:
                if step > 0:
                    average_loss /= print_freq
                # The average loss is an estimate of the loss over the last 2000 batches.
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0
                saver.save(sess, save_dir, global_step=global_step)

            if step % eval_freq == 0 or step == num_steps-1:
                train_loss, train_loss_sum, global_step_val = sess.run(
                    [full_loss, train_full_loss_summary, global_step],
                    feed_dict=train_eval_feed_dict)
                test_loss, test_loss_sum = sess.run([full_loss, test_full_loss_summary],
                                                     feed_dict=test_eval_feed_dict)
                writer.add_summary(train_loss_sum, global_step)
                writer.add_summary(test_loss_sum, global_step)
                print("Evaluated loss at step {}: train={:.3f} test={:.3f}".format(
                    global_step_val, train_loss, test_loss))

if __name__ == '__main__':
    train()
