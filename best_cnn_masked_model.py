import numpy as np
from random import sample
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.model_selection import KFold

RUN_IDENTIFIER = 'best_cnn_masked_01'
SEED_VALUE = 12345


def import_data():
    bip_address = 'data/data_mask_SH_B/data_mask_SH_B_bip.npy'
    shiz_address = 'data/data_mask_N_/data_mask_SH_B_shiz.npy'

    data_bip = np.load(bip_address)
    # data_bip = data_bip[0:5,:,:,:]

    data_shiz = np.load(shiz_address)
    # data_shiz = data_shiz[0:5,:,:,:]

    # assigning zero label to N_Bs
    labels_bip = np.zeros(data_bip.shape[0])

    # assigning one label to N_SHs
    labels_shiz = np.ones(data_shiz.shape[0])

    # concatenating both groups together
    data_together = np.concatenate((data_bip, data_shiz), axis=0)
    labels_together = np.concatenate((labels_bip, labels_shiz), axis=0)

    # data_together = data_together[:, :, np.reshape(candidate_cols == 0, -1)]
    return data_together, labels_together


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 7))
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

    kf_manager = KFold(n_splits=inputs.shape[0], shuffle=True, random_state=2)
    k_fold_counter = 0
    test_accuracies = []

    for train_index, test_index in kf_manager.split(inputs):
        k_fold_counter += 1
        best_model = keras.models.load_model(f'model_{k_fold_counter}')
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
