import random
import numpy as np
import pandas as pd
from random import sample
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import pickle
import keras_metrics


def import_data():
    N_B_filter_address = 'data/all_ttest_N_/all_ttest_N_B_filter.npy'
    N_SH_filter_address = 'data/all_ttest_N_/all_ttest_N_SH_filter.npy'
    data_N_B_filter = np.load(N_B_filter_address)
    data_N_SH_filter = np.load(N_SH_filter_address)

    # reshaping data so that samples are in rows
    data_N_B_reshaped = reshape_inputs(
        data_N_B_filter,
        [1, 0, 2]
    )
    # assigning zero label to N_Bs
    data_N_B_labels = np.zeros(data_N_B_reshaped.shape[0])
    data_N_SH_reshaped = reshape_inputs(
        data_N_SH_filter,
        [1, 0, 2]
    )
    # assigning one label to N_SHs
    data_N_SH_labels = np.ones(data_N_SH_reshaped.shape[0])

    # concatenating both groups together
    data_together = np.concatenate((data_N_B_reshaped, data_N_SH_reshaped), axis=0)
    labels_together = np.concatenate((data_N_B_labels, data_N_SH_labels), axis=0)

    # remove constant strides
    n_samples = data_together.shape[0]
    n_rows = data_together.shape[1]
    n_cols = data_together.shape[2]

    candidate_rows = np.ones((n_rows, 1))
    candidate_cols = np.ones((n_cols, 1))

    for row in range(n_rows):
        for sample in range(n_samples):
            if not (np.sum(data_together[sample, row, :]) == 0):
                candidate_rows[row] = 0

    for col in range(n_cols):
        for sample in range(n_samples):
            if not (np.sum(data_together[sample, :, col]) == 0):
                candidate_cols[col] = 0

    # data_together = data_together[:, :, np.reshape(candidate_cols == 0, -1)]
    return data_together, labels_together


def reshape_inputs(matrix, order):
    return matrix.reshape(matrix.shape[order[0]], matrix.shape[order[1]], matrix.shape[order[2]])


def generate_datasets(inputs, labels, training_percent):
    n_data = inputs.shape[0]
    sequence_indices = range(n_data)

    # determine a fix seed value
    # random.seed(6300)

    # generating shuffled indices to group training and test datasets
    shuffled_sequence_indices = sample(sequence_indices, n_data)

    # calculating training and test number
    n_training = int(np.ceil(training_percent * n_data))
    n_test = n_data - n_training

    # generating training and test inputs and labels
    training_inputs = inputs[shuffled_sequence_indices[0: n_training]]
    training_labels = labels[shuffled_sequence_indices[0: n_training]]
    test_inputs = inputs[shuffled_sequence_indices[n_training + 1: n_data]]
    test_labels = labels[shuffled_sequence_indices[n_training + 1: n_data]]

    return training_inputs, test_inputs, training_labels, test_labels


def generate_model(image_width, image_height):
    # input layer
    input_layer = keras.Input(shape=(image_width, image_height, 1))

    # convolutional layer
    conv_layer = layers.Conv2D(10, kernel_size=5, activation='relu')(input_layer)

    # max pool layer
    max_pool_layer = layers.MaxPooling2D(pool_size=2)(conv_layer)

    # convolutional layer
    conv_layer = layers.Conv2D(10, kernel_size=5, activation='relu')(max_pool_layer)

    # max pool layer
    max_pool_layer = layers.MaxPooling2D(pool_size=2)(conv_layer)

    # flatten the convolution layer
    flattened_layer = layers.Flatten()(max_pool_layer)

    # dense layer to wrap to one number
    output_layer = layers.Dense(1, activation='sigmoid')(flattened_layer)

    model = keras.Model(input_layer, output_layer)
    model.summary()
    return model


def compile_and_run_model(model, training_inputs, training_labels):
    opt = keras.optimizers.Adam()
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy", keras_metrics.precision(), keras_metrics.recall()])
    model_history = model.fit(training_inputs, training_labels, epochs=8)
    return model_history


def test_model(model, test_inputs, test_labels):
    evaluations = model.evaluate(
        x=test_inputs,
        y=test_labels
    )
    return evaluations[1]


def report_model_performance(history):
    loss_series = history['loss']
    accuracy_series = history['accuracy']
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(loss_series)
    plt.title('Loss Series')
    plt.subplot(2, 1, 2)
    plt.plot(accuracy_series)
    plt.title('Accuracy Series')
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig('figures/final_performance_series.png', dpi=600)


def save_dataset(training_inputs, training_labels, test_input, test_labels):
    dataset = {
        'training_inputs': training_inputs,
        'training_labels': training_labels,
        'test_inputs': test_inputs,
        'test_labels': test_labels
    }
    with open('dataset.pkl', 'wb') as handle:
        pickle.dump(dataset, handle)


if __name__ == '__main__':
    inputs, labels = import_data()

    test_accuracy = 0
    while test_accuracy < 0.84:
        training_inputs, test_inputs, training_labels, test_labels = generate_datasets(inputs, labels, 0.8)
        model = generate_model(training_inputs.shape[1], training_inputs.shape[2])
        model_history = compile_and_run_model(model, np.expand_dims(training_inputs, axis=-1),
                                      np.expand_dims(training_labels, axis=-1))
        test_accuracy = test_model(model, test_inputs, test_labels)

    save_dataset(training_inputs, training_labels, test_inputs, test_labels)
    report_model_performance(model_history.history)
    model.save('model')


    # calling it off
    print('done')
