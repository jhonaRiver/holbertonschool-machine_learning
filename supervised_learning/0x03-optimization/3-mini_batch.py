#!/usr/bin/env python3
"""
module train_mini_batch
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    trains a loaded neural network model using mini-batch gradient descent
    Args:
        X_train (ndarray): contains the training data
        Y_train (ndarray): contains the training labels
        X_valid (ndarray): contains the validation data
        Y_valid (ndarray): contains the validation labels
        batch_size (int, optional): number of data points in a batch. Defaults
                                    to 32.
        epochs (int, optional): number of times the training should pass
                                through the whole dataset. Defaults to 5.
        load_path (str, optional): path from which to load the model. Defaults
                                   to "/tmp/model.ckpt".
        save_path (str, optional): path to where the model should be saved
                                   after training. Defaults to
                                   "/tmp/model.ckpt".
    Returns:
        path where the model was saved
    """
    with tf.Session() as sess:
        save_NN = tf.train.import_meta_graph("{}.meta".format(load_path))
        save_NN.restore(sess, load_path)
        graph = tf.get_default_graph()
        m = X_train.shape[0]
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]
        for epoch in range(epochs+1):
            steps = m // batch_size + 1
            train_cost = sess.run(loss, feed_dict={x: X_train,
                                                   y: Y_train})
            train_accuracy = sess.run(accuracy, feed_dict={x: X_train,
                                                           y: Y_train})
            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid,
                                                           y: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))
            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
            for step_number in range(steps):
                start = batch_size * step_number
                end = batch_size * (step_number + 1)
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                if (step_number + 1) % 100 == 0:
                    step_cost = sess.run(loss, feed_dict={x: X_batch,
                                                          y: Y_batch})
                    step_accuracy = sess.run(accuracy, feed_dict={x: X_batch,
                                                                  y: Y_batch})
                    print("\tStep {}:".format(step_number + 1))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))
        return save_NN.save(sess, save_path)
