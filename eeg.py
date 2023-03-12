import numpy as np
import pandas as pd
import mne
import os
import re
from scipy.io import loadmat

#%%
# read raw EEG dataset of .gdf file
class RawEEGData:
    # rawData = list()
    # event = pd.DataFrame()
    # channel = list()
    # sample_freq = 0

    def __init__(self, file_name):
        raw = mne.io.read_raw_gdf(file_name, preload=True, stim_channel=-1)
        event_type2idx = {276: 0, 277: 1, 768: 2, 769: 3, 770: 4, 781: 5, 783: 6, 1023: 7, 1077: 8, 1078: 9, 1079: 10,
                          1081: 11, 32766: 12}
        self.rawData = raw._data
        self.channel = raw._raw_extras[0]['ch_names']
        self.sample_freq = raw.info['sfreq']
        # self.rawData =
        self.event = pd.DataFrame({
            "length":raw._raw_extras[0]['events'][0],
            "position": raw._raw_extras[0]['events'][1],
            "event type": raw._raw_extras[0]['events'][2],
            "event index": [event_type2idx[event_type] for event_type in raw._raw_extras[0]['events'][2]],
            "duration": raw._raw_extras[0]['events'][4],
            "CHN": raw._raw_extras[0]['events'][3]
        })

    # print event type information of EEG data set
    @staticmethod
    def print_type_info():
        print("EEG data set event information and index:")
        print("%12s\t%10s\t%30s" % ("Event Type", "Type Index", "Description"))
        print("%12d\t%10d\t%30s" % (276, 0, "Idling EEG (eyes open)"))
        print("%12d\t%10d\t%30s" % (277, 1, "Idling EEG (eyes closed"))
        print("%12d\t%10d\t%30s" % (768, 2, "Start of a trial"))
        print("%12d\t%10d\t%30s" % (769, 3, "Cue onset left (class 1)"))
        print("%12d\t%10d\t%30s" % (770, 4, "Cue onset right (class 2)"))
        print("%12d\t%10d\t%30s" % (781, 5, "BCI feedback (continuous"))
        print("%12d\t%10d\t%30s" % (783, 6, "Cue unknown"))
        print("%12d\t%10d\t%30s" % (1023, 7, "Rejected trial"))
        print("%12d\t%10d\t%30s" % (1077, 8, "Horizontal eye movement"))
        print("%12d\t%10d\t%30s" % (1078, 9, "Vertical eye movement"))
        print("%12d\t%10d\t%30s" % (1079, 10, "Eye rotation"))
        print("%12d\t%10d\t%30s" % (1081, 11, "Eye blinks"))
        print("%12d\t%10d\t%30s" % (32766, 12, "Start of a new run"))

#%%
# arrange data for training and test
def get_data(data_file_dir, labels_file_dir):
    RawEEGData.print_type_info()
    sfreq = 250  # sample frequency of dataset
    # read data file
    data = dict()
    data_files = os.listdir(data_file_dir)
    for data_file in data_files:
        if not re.search(".*\.gdf", data_file):
            continue
        
        info = re.findall('B0([0-9])0([0-9])[TE]\.gdf', data_file)
        try:
            subject = "subject" + info[0][0]
            session = "session" + info[0][1]
            filename = data_file_dir + "\\" + data_file
            print(filename)
            raw_eeg_data = RawEEGData(filename)
            trial_event = raw_eeg_data.event[raw_eeg_data.event['event index'] == 2]
            session_data = dict()
            for event, event_data in trial_event.iterrows():
                trial_data = raw_eeg_data.rawData[:, event_data['position']:event_data['position']+event_data['duration']]
                for idx in range(len(raw_eeg_data.channel)):
                    if raw_eeg_data.channel[idx] not in session_data:
                        session_data[raw_eeg_data.channel[idx]] = list()
                    session_data[raw_eeg_data.channel[idx]].append(trial_data[idx])
            if subject not in data:
                data[subject] = dict()
            data[subject][session] = session_data
        except Exception as e:
            print(e)
#            raise ("invalid data file name")

    # read data file
    labels = dict()
    labels_files = os.listdir(labels_file_dir)
    for labels_file in labels_files:
        if not re.search(".*\.mat", labels_file):
            continue

        info = re.findall('B0([0-9])0([0-9])[TE]\.mat', labels_file)
        try:
            subject = "subject" + info[0][0]
            session = "session" + info[0][1]
            filename = labels_file_dir + "/" + labels_file
            print(filename)
            session_label = loadmat(filename)
            session_label = session_label['classlabel'].astype(np.int8)
            if subject not in labels:
                labels[subject] = dict()
            labels[subject][session] = session_label
        except Exception as e:
            print(e)
            raise ("invalid labels file name")

    return data, labels, sfreq
    # print(data)
    # print(labels)
#%%
#%%
# arrange data for training and test
def get_data(data_file_dir, labels_file_dir):
    RawEEGData.print_type_info()
    sfreq = 250  # sample frequency of dataset
    # read data file
    data = dict()
    data_files = os.listdir(data_file_dir)
    for data_file in data_files:
        if not re.search(".*\.gdf", data_file):
            continue
        
        info = re.findall('B0([0-9])0([0-9])[TE]\.gdf', data_file)
        try:
            subject = "subject" + info[0][0]
            session = "session" + info[0][1]
            filename = data_file_dir + "/" + data_file
            print(filename)
            raw_eeg_data = RawEEGData(filename)
            trial_event = raw_eeg_data.event[raw_eeg_data.event['event index'] == 2]
            session_data = dict()
            for event, event_data in trial_event.iterrows():
                trial_data = raw_eeg_data.rawData[:, event_data['position']:event_data['position']+event_data['duration']]
                for idx in range(len(raw_eeg_data.channel)):
                    if raw_eeg_data.channel[idx] not in session_data:
                        session_data[raw_eeg_data.channel[idx]] = list()
                    session_data[raw_eeg_data.channel[idx]].append(trial_data[idx])
            if subject not in data:
                data[subject] = dict()
            data[subject][session] = session_data
        except Exception as e:
            print(e)
#            raise ("invalid data file name")

    

    return data, sfreq
    # print(data)
    # print(labels)

#%%
if __name__ == "__main__":
    # for test 1
    filename = "/Users/aytijhyasaha/Documents/datasets/sensor-selection-datasets/BCICIV_2b_gdf/B0101T.gdf"
    d = RawEEGData(filename)
    data = d.event[d.event['event index'] == 2]
    # print(0, "\n", data.shape)

    data_src = "/Users/aytijhyasaha/Documents/datasets/sensor-selection-datasets/BCICIV_2b_gdf"
    labels_src = "true_labels"
    data, sfreq = get_data(data_src, labels_src)
    print("test: read dataset")
    
#%%
import numpy as np
from sklearn.model_selection import KFold
import pickle
import tensorflow as tf
import keras
import pandas as pd


from signalProcessing import run_sig_processing

# get train data and labels for each segment
def arrange_data(data, labels):
    output_data = list()
    output_labels = list()
    for idx in range(len(data)):
        for segment in data['subject'+str(idx+1)]:
            output_data.append(np.expand_dims(segment, axis=2))
            if labels[idx][0] == 1:
                output_labels.append(0)
            else:
                output_labels.append(1)
    output_data = np.array(output_data)
    output_labels = np.array(output_labels)
    return output_data, output_labels


# build model    4-1
# def build_model(size_y, size_x):
#     # input layer
#     img_input = keras.layers.Input(shape=(size_y, size_x, 1))
#
#     # First convolution extracts 30 filters that are (size_y, 3)
#     # Convolution is followed by max-pooling layer with a 1x10 window
#     x = keras.layers.Conv2D(filters=30, kernel_size=(size_y, 3), activation='relu')(img_input)
#     x = keras.layers.MaxPooling2D(1, 10)(x)
#
#     # Flatten feature map to a 1-dim tensor so we can add fully connected layers
#     x = keras.layers.Flatten()(x)
#
#     # Create a fully connected layer with ReLU activation and 512 hidden units
#     # x = keras.layers.Dense(512, activation='relu')(x)
#
#     # Add a dropout rate of 0.5
#     # x = keras.layers.Dropout(0.75)(x)
#
#     # Create output layer with a single node and sigmoid activation
#     output = keras.layers.Dense(2, activation='sigmoid')(x)
#
#     # Create model:
#     # input = input feature map
#     # output = input feature map + stacked convolution/maxpooling layers + fully
#     # connected layer + sigmoid output layer
#     model = keras.models.Model(img_input, output)
#     model.summary()
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=keras.optimizers.RMSprop(lr=0.001),
#                   metrics=['acc'])
#     return model

def build_model(size_y, size_x, dim=1):
    # input layer
    img_input = keras.layers.Input(shape=(size_y, size_x, dim))

    # First convolution extracts 30 filters that are (size_y, 3)
    # Convolution is followed by max-pooling layer with a 1x10 window
    x = keras.layers.Conv2D(filters=30, kernel_size=(size_y, 3), activation='relu',
                            kernel_regularizer=keras.regularizers.l2(0))(img_input)
    x = keras.layers.MaxPooling2D(1, 10)(x)

    # Convolution is followed by max-pooling layer with a 1x10 window
    # x = keras.layers.Conv2D(filters=10, kernel_size=(1, 5), activation='relu',
    #                         kernel_regularizer=keras.regularizers.l2(0.05))(x)
    # x = keras.layers.MaxPooling2D(1, 3)(x)

    # Flatten feature map to a 1-dim tensor so we can add fully connected layers
    x = keras.layers.Flatten()(x)

    # Create a fully connected layer with ReLU activation and 512 hidden units
    x = keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    # x = keras.layers.MaxPooling1D(2)(x)

    # Create a fully connected layer with ReLU activation and 512 hidden units
    # x = keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)

    # Add a dropout rate of 0.5
    # x = keras.layers.Dropout(0.75)(x)

    # Create a fully connected layer with ReLU activation and 512 hidden units
    # x = keras.layers.Dense(256, activation='sigmoid')(x)

    # Add a dropout rate of 0.5
    # x = keras.layers.Dropout(0.75)(x)

    # Create output layer with a single node and sigmoid activation
    output = keras.layers.Dense(2, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(0))(x)

    # Create model:
    # input = input feature map
    # output = input feature map + stacked convolution/maxpooling layers + fully
    # connected layer + sigmoid output layer
    model = keras.models.Model(img_input, output)
    model.summary()
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=keras.optimizers.RMSprop(lr=0.001),
    #               metrics=['acc'])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model


# evaluated trial to trial performance
def trial_evaluate(model, data, labels):
    acc = 0.0
    for idx in range(len(data)):
        test_data, test_label = arrange_data(np.expand_dims(data[idx], axis=0), np.expand_dims(labels[idx], axis=0))
        test_label = keras.utils.to_categorical(test_label, num_classes=2)
        loss, accuracy = model.evaluate(test_data, test_label)
        if accuracy > 0.5:
            acc += 1.0
    acc = acc/len(data)
    return acc


# run classification
def run_classification(data, labels, session=(1,2,3,4,5)):
    kf = KFold(n_splits=10, shuffle=True)
    classification_acc = pd.DataFrame()
    for subject in data:
        # if subject == 'subject1': continue
        subject_acc = list()
        input_data = list()
        target_labels = list()
        # combine trials data of target session
        [input_data.extend(data[subject]["session" + str(idx)]['input data']) for idx in session]
        [target_labels.extend(labels[subject]["session" + str(idx)]) for idx in session]
        input_data = np.array(input_data)
        target_labels = np.array(target_labels)

        # 10 fold cross-validation
        count = 0
        for train_index, test_index in kf.split(input_data):
            count += 1
            train_data, train_labels = arrange_data(input_data[train_index], target_labels[train_index])
            test_data, test_labels = arrange_data(input_data[test_index], target_labels[test_index])

            size_y, size_x = train_data[0].shape[0:2]

            print(train_data.shape)
            # train_data_size = train_data.shape[0]
            # test_data_size = test_data.shape[0]

            train_labels = keras.utils.to_categorical(train_labels, num_classes=2)
            test_labels = keras.utils.to_categorical(test_labels, num_classes=2)


            # build model 
            model = build_model(size_y, size_x)

            print('Training ------------')
            # train the model
            model.fit(train_data, train_labels, epochs=300, batch_size=40)

            print('\nTesting ------------')
            # Evaluate the model with the metrics we defined earlier
            loss, accuracy = model.evaluate(test_data, test_labels)

            trial_acc = trial_evaluate(model, input_data[test_index], target_labels[test_index])
            print(count, subject)
            print('test loss: ', loss)
            print('test accuracy: ', accuracy)
            print('trial to trial accuracy: ', trial_acc)
            subject_acc.append(trial_acc)
        classification_acc[subject] = subject_acc
    return classification_acc




if __name__ == '__main__':
    # '''
    data_src = r"BCICIV_2b_gdf"
    labels_src = r"true_labels"

    #
    # band_type: 0: band pass feature, 1: AR PSD feature, 2: extend band
    #
    # data, labels = run_sig_processing(data_src, labels_src, band_type=2)

    # Saving the data and labels:
    # with open('temp_data.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([data, labels], f)
    # '''
    # Getting back the data and labels:
    with open('temp_data.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        data, labels = pickle.load(f)

    res = run_classification(data, labels)
    print(res)
    res.to_csv("BP_acc.csv", encoding="utf-8")
    print("cnn classification")
    