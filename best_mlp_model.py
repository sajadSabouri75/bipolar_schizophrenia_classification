import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.model_selection import KFold

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


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    plt.figure(figsize = (10,7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def test_model(model, test_inputs, test_labels):
    evaluations = model.evaluate(
        x=test_inputs,
        y=test_labels
    )
    return evaluations[1]


if __name__ == '__main__':
    inputs, labels = import_data()

    kf_manager = KFold(n_splits=5, shuffle=True, random_state=SEED_VALUE)
    k_fold_counter = 0
    test_accuracies = []

    for train_index, test_index in kf_manager.split(inputs):
        k_fold_counter += 1
        best_model = keras.models.load_model(f'outputs/{RUN_IDENTIFIER}_{k_fold_counter}')
        print(k_fold_counter)
        training_inputs = inputs[train_index]
        test_inputs = inputs[test_index]
        training_labels = labels[train_index]
        test_labels = labels[test_index]

        # Predict the values from the validation dataset
        test_predictions = best_model.predict(test_inputs)

        # Convert predictions classes from one hot vectors to labels: [0 0 1 0 0 ...] --> 2
        test_predictions = np.round(test_predictions)

        confusion_mtx = confusion_matrix(test_labels, test_predictions)

        # plot the confusion matrix
        plot_confusion_matrix(confusion_mtx, classes=['Bipolar', 'Schizophrenia'])

        test_accuracies.append(test_model(best_model, test_inputs, test_labels))

    print(test_accuracies)
    print(f'mean of test accuracies = {np.mean(np.array(test_accuracies))}')




