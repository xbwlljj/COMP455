# Samen Sethi 300157474
# Bowen Xue 300139793

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer


# Load the label names from file
def _load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# Load a batch of the CIFAR-10 dataset
def load_cfar10_batch(cifar10_dataset_folder_path, batchId):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batchId), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels


# Display Stats of the the dataset
def display_stats(cifar10_dataset_folder_path, batchId, sampleId):
    batch_ids = list(range(1, 6))

    if batchId not in batch_ids:
        print('Batch Id out of Range. Batch Ids should be: {}'.format(batch_ids))
        return None

    features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batchId)

    if not (0 <= sampleId < len(features)):
        print('{} samples in batch {}.  {} is out of range.'.format(len(features), batchId, sampleId))
        return None

    print('First 20 Labels: {}'.format(labels[:20]))
    print('\nStats of batch {}:'.format(batchId))
    print('Label Counts: {}'.format(dict(zip(*np.unique(labels, return_counts=True)))))
    print('Samples: {}'.format(len(features)))


    sampleImage = features[sampleId]
    sampleLabel = labels[sampleId]
    labelNames = _load_label_names()

    print('\nImage example {}:'.format(sampleId))
    print('Image - Max: {} Min: {}'.format(sampleImage.max(), sampleImage.min()))
    print('Image - Shape: {}'.format(sampleImage.shape))
    print('Label - Id of label: {} Name of label: {}'.format(sampleLabel, labelNames[sampleLabel]))
    plt.axis('off')
    plt.imshow(sampleImage)


# Save the data to file before processing
def _preprocess_and_save(normal, one_hot_encode, features, labels, filename):
    features = normal(features)
    labels = one_hot_encode(labels)
    pickle.dump((features, labels), open(filename, 'wb'))


# Training and Validation Data before processing
def preprocess_and_save_data(cifar10_dataset_folder_path, normal, one_hot_encode):

    n_batches = 5
    valid_features = []
    valid_labels = []

    for batch_i in range(1, n_batches + 1):
        features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)
        validation_count = int(len(features) * 0.1)

        # Pre-process and save a batch of training data
        _preprocess_and_save(
            normal,
            one_hot_encode,
            features[:-validation_count],
            labels[:-validation_count],
            'pre-process_batch_' + str(batch_i) + '.p')

        # Use a portion of training batch for validation
        valid_features.extend(features[-validation_count:])
        valid_labels.extend(labels[-validation_count:])

    # Pre-process and Save all validation data
    _preprocess_and_save(
        normal,
        one_hot_encode,
        np.array(valid_features),
        np.array(valid_labels),
        'pre-process_validation.p')

    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # load the training data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    # Pre-process and Save all training data
    _preprocess_and_save(
        normal,
        one_hot_encode,
        np.array(test_features),
        np.array(test_labels),
        'pre-process_training.p')


# Separate features and labels into batches
def batch_features_labels(features, labels, batch_size):

    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]


# Load the Pre-processed training data
def load_preprocess_training_batch(batchId, batchSize):
    filename = 'preprocess_batch_' + str(batchId) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size>
    return batch_features_labels(features, labels, batchSize)


# Display the image based on the feature, labels, and predictions
def display_image_predictions(features, labels, predictions):
    n_classes = 10
    label_names = _load_label_names()
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(n_classes))
    label_ids = label_binarizer.inverse_transform(np.array(labels))

    fig, axies = plt.subplots(nrows=4, ncols=2)
    fig.tight_layout()
    fig.suptitle('Softmax Predictions', fontsize=20, y=1.1)

    n_predictions = 3
    margin = 0.05
    ind = np.arange(n_predictions)
    width = (1. - 2. * margin) / n_predictions

    for image_i, (feature, label_id, pred_indicies, pred_values) in enumerate(zip(features, label_ids, predictions.indices, predictions.values)):
        pred_names = [label_names[pred_i] for pred_i in pred_indicies]
        correct_name = label_names[label_id]

        axies[image_i][0].imshow(feature*255)
        axies[image_i][0].set_title(correct_name)
        axies[image_i][0].set_axis_off()

        axies[image_i][1].barh(ind + margin, pred_values[::-1], width)
        axies[image_i][1].set_yticks(ind + margin)
        axies[image_i][1].set_yticklabels(pred_names[::-1])
        axies[image_i][1].set_xticks([0, 0.5, 1.0])
