from sklearn.model_selection import KFold
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

RUN_IDENTIFIER = 'mlp_03'
SEED_VALUE = 12345


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


def generate_model():
    # input layer
    input_layer = keras.Input(shape=(192, 192, 1))

    # flatten the convolution layer
    flattened_layer = layers.Flatten()(input_layer)

    # hidden dense layer
    hidden_layer = layers.Dense(
        units=10,
        activation='sigmoid'
    )(flattened_layer)

    # hidden dense layer
    hidden_layer = layers.Dense(
        units=10,
        activation='sigmoid'
    )(hidden_layer)

    # dense layer to wrap to one number
    output_layer = layers.Dense(1, activation='sigmoid')(hidden_layer)

    model = keras.Model(input_layer, output_layer)

    model.summary()

    return model


def compile_and_run_model(model, training_inputs, training_labels):
    opt = keras.optimizers.Adam()
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
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
    plt.figure(counter)
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

    model = generate_model()

    kf_manager = KFold(n_splits=5, shuffle=True, random_state=SEED_VALUE)
    k_fold_counter = 0
    test_accuracies = []
    for train_index, test_index in kf_manager.split(inputs):
        k_fold_counter += 1
        print(k_fold_counter)
        training_inputs = inputs[train_index]
        test_inputs = inputs[test_index]
        training_labels = labels[train_index]
        test_labels = labels[test_index]
        model_history = compile_and_run_model(
            model,
            training_inputs,
            training_labels
        )
        test_accuracies.append(test_model(model, test_inputs, test_labels))
        model.save(f'outputs/{RUN_IDENTIFIER}_{k_fold_counter}')
        report_model_performance(model_history.history, k_fold_counter)

    print(test_accuracies)
    print(f'mean of test accuracies = {np.mean(np.array(test_accuracies))}')

    # calling it off
    print('done')
