from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from random import sample
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import keras_tuner
from scipy.sparse import csr_matrix


RUN_IDENTIFIER = 'cnn_masked_01'
SEED_VALUE = 12345


def import_data():
    bip_address = 'data/data_mask_SH_B/data_mask_SH_B_bip.npy'
    shiz_address = 'data/data_mask_N_/data_mask_SH_B_shiz.npy'

    data_bip = np.load(bip_address)

    data_shiz = np.load(shiz_address)

    # assigning zero label to N_Bs
    labels_bip = np.zeros(data_bip.shape[0])

    # assigning one label to N_SHs
    labels_shiz = np.ones(data_shiz.shape[0])

    # concatenating both groups together
    data_together = np.concatenate((data_bip, data_shiz), axis=0)
    labels_together = np.concatenate((labels_bip, labels_shiz), axis=0)

    # data_together = data_together[:, :, np.reshape(candidate_cols == 0, -1)]
    return data_together, labels_together


def generate_model():
    # input layer
    input_layer = keras.Input(shape=(229, 192, 192, 1))

    # convolutional layer
    conv_layer = layers.Conv3D(
        10,
        kernel_size=3,
        activation='relu'
    )(input_layer)

    # max pool layer
    max_pool_layer = layers.MaxPooling3D(
        pool_size=3
    )(conv_layer)

    # convolutional layer
    conv_layer = layers.Conv3D(
        filters=10,
        kernel_size=3,
        activation='relu'
    )(max_pool_layer)

    # max pool layer
    max_pool_layer = layers.MaxPooling3D(
        pool_size=3
    )(conv_layer)

    # flatten the convolution layer
    flattened_layer = layers.Flatten()(max_pool_layer)

    # hidden dense layer
    hidden_layer = layers.Dense(
        units=10,
        activation='sigmoid'
    )(flattened_layer)

    # dense layer to wrap to one number
    output_layer = layers.Dense(1, activation='sigmoid')(hidden_layer)

    model = keras.Model(input_layer, output_layer)

    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4]),
    #     decay_steps=10000,
    #     decay_rate=0.9
    # )
    opt = keras.optimizers.Adam()
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    model.summary()

    return model


def compile_and_run_model(model, training_inputs, training_labels):
    model_history = model.fit(training_inputs, training_labels, epochs=8)
    return model_history


def test_model(model, test_inputs, test_labels):
    evaluations = model.evaluate(
        x=test_inputs,
        y=test_labels
    )
    return evaluations[1]


def report_model_performance(history, counter):
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
    # plt.show(block=False)
    plt.savefig(f'outputs/figures/{RUN_IDENTIFIER}_{counter}.png', dpi=400)


if __name__ == '__main__':
    inputs, labels = import_data()

    kf_manager = KFold(n_splits=inputs.shape[0], shuffle=True, random_state=SEED_VALUE)
    test_accuracies = []

    model = generate_model()

    k_fold_counter = 0
    for train_index, test_index in kf_manager.split(inputs):
        k_fold_counter += 1

        print(f'> Running Fold {str(k_fold_counter).zfill(2)}')

        training_inputs = inputs[train_index]
        test_inputs = inputs[test_index]
        training_labels = labels[train_index]
        test_labels = labels[test_index]
        model_history = compile_and_run_model(
            model,
            np.expand_dims(training_inputs,-1),
            np.expand_dims(training_labels,-1),
        )
        test_accuracies.append(test_model(model, test_inputs, test_labels))
        model.save(f'outputs/{RUN_IDENTIFIER}_{k_fold_counter}')
        report_model_performance(model_history.history, k_fold_counter)

    print(test_accuracies)

    # calling it off
    print('done')
