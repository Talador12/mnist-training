import os
import sys
import tensorflow as tensorflow
import numpy as np
import matplotlib.pyplot as plt

tf.compat.v1.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
FLAGS = tf.compat.v1.app.flags.FLAGS

def get_data():
    """
    Helper function to get mnist data, flattening that data, and assigning it to  a single dictionary for return
    Arguments:
        None
    Return:
        [Dict] -- Dictionary containing the test and train, data and labels for each
    """
    mnist = tf.keras.datasets.mnist
    (
        train_images_2d,
        train_labels_n
    ),
    (
        test_images_2d,
        test_labels_n
    ) = mnist.load_data()
    data = {
        'train_data': train_images_2d.rehsape(train_images_2d.shape[0], 784).astype('float32') / 255,
        'train_labels': tf.keras.utils.to_categorical(train_labels_n, 10),
        'test_data': test_images_2d.reshape(test_images_2d.shape[0], 784).astype('float32') / 255,
        'test_labels': tf.keras.utils.to_categorical(test_labels_n, 10)
    }

    return data


def display_one_hot_sample(num, train_data, train_labels):
    """
    Helper function to display a visual of one sample number chosen by `num`

    Arguments:
        num - [Integer] -- number classification to find in the data and display
        train_data - [Array] -- Array of all flattened train data
        train_labels - [Array] -- Array of all categorical train labels
    Return:
        None
    """
    label = train_labels[num].argmax(axis=0)
    image = train_data[num].reshape([28, 28])
    plt.title(f'Sample: {num} Label: {label}')
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))

    plt.show()


def display_epoch_image(train_data):
    """
    Helper function to display the first 500 training samples

    Arguments:
        train_data - [Array] -- Array of all flattened train data
    Return:
        None
    """
    images = train_data[0].reshape([1, 784])
    for i in range(1, 500):
        images = np.concatenate((images, train_data[i].reshape([1, 784])))
    plt.imshow(images, cmap=plt.get_cmap('gray_r'))

    plt.show()


def next_batch(num, data, labels):
    """
    """
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def train_model(train_data, train_labels, test_data, test_labels):
    """
    Trains the model given the data and labels for train and test respectively

    Arguments:
        train_data - [Array] -- Array of all flattened train data
        train_labels - [Array] -- Array of all categorical train labels
        test_data - [Array] -- Array of all flattened test data
        test_labels - [Array] -- Array of all categorical test labels
    """
    # initialize inputs
    input_images = tf.placeholder(tf.float32, shape=[None, 784])
    target_labels = tf.placeholder(tf.float32, shape=[None, 10])

    # initialize tensorflow weights and biases
    hidden_nodes = 512
    input_weights = tf.Variable(tf.truncated_normal([784, hidden_nodes]))
    input_biases = tf.Variable(tf.Variable(tf.zeros([hidden_nodes])))
    hidden_weights = tf.Variable(tf.truncated_normal([hidden_nodes, 10]))
    hidden_biases = tf.Variable(tf.zeros([10]))

    # setup layers, weights, and optimizer for the neural net
    input_layer = tf.matmul(input_images, input_weights)
    hidden_layer = tf.nn.relu(input_layer + input_biases)
    digit_weights = tf.matmul(hidden_layer, hidden_weights) + hidden_biases
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=digit_weights, labels=target_labels))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # setup to measure accuracy
    correct_prediction = tf.equal(tf.argmax(digit_weights, 1), tf.argmax(target_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # start session for training
    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
        sess.run(tf.global_variables_intializer())
        # For 2000 epochs
        for x in range(2000):
            # Run in batches of 100 epochs
            (x_batch, y_batch) = next_batch(100, train_data, train_labels)
            sess.run(optimizer, feed_dict={input_images: x_batch, target_labels: y_batch})
            # Access accuracy each 100 epochs
            if(x % 100):
                print(f'Training epoch {x+1}')
                print(f'Accuracy: {sess.run(accuracy, feed_dict={input_images: test_data, target_labels: test_labels})}')


def export_model(train_data, train_labels):
    """
    Exports a trained model given the data and labels for train set

    Arguments:
        train_data - [Array] -- Array of all flattened train data
        train_labels - [Array] -- Array of all categorical train labels
    """
    export_path_base = sys.argv[-1]
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(FLAGS.model_version)))
    print('Exporting trained model to', export_path)
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_path)

    # Build the signature_def_map.
    classification_inputs = tf.compat.v1.saved_model.utils.build_tensor_info(serialized_tf_example)
    classification_outputs_classes = tf.compat.v1.saved_model.utils.build_tensor_info(prediction_classes)
    classification_outputs_scores = tf.compat.v1.saved_model.utils.build_tensor_info(values)

    classification_signature = (
        tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
            inputs={
                tf.compat.v1.saved_model.signature_constants.CLASSIFY_INPUTS:
                    classification_inputs
            },
            outputs={
                tf.compat.v1.saved_model.signature_constants
                .CLASSIFY_OUTPUT_CLASSES:
                    classification_outputs_classes,
                tf.compat.v1.saved_model.signature_constants
                .CLASSIFY_OUTPUT_SCORES:
                    classification_outputs_scores
            },
            method_name=tf.compat.v1.saved_model.signature_constants
            .CLASSIFY_METHOD_NAME))

    tensor_info_x = tf.compat.v1.saved_model.utils.build_tensor_info(train_data)
    tensor_info_y = tf.compat.v1.saved_model.utils.build_tensor_info(train_labels)

    prediction_signature = (
        tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': tensor_info_x},
            outputs={'scores': tensor_info_y},
            method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
        sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_images':
                prediction_signature,
            tf.compat.v1.saved_model.signature_constants
            .DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                classification_signature,
        },
        main_op=tf.compat.v1.tables_initializer(),
        strip_default_attrs=True)

    builder.save()

    print('Done exporting!')


if __name__ == '__main__':
    data = get_data()
    display_one_hot_sample(100, data['train_data'], data['train_labels'])
    display_epoch_image(data['train_data'])
    train_model(data['train_data'], data['train_labels'], data['test_data'], data['test_labels'])
    export_model(data['train_data'], data['train_labels'])
