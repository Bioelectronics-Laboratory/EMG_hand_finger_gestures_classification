# import libraries

import os
import pandas as pd
import numpy as np
from datetime import datetime

import imblearn

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#############################################################################
'''
Here, the datasets we used in the paper can not be released for personal information protection.
Instead, you can identify a sample dataset.
please refer to "sample_dataset.csv"
'''

# Load dataset
data = pd.read_csv("your_own_dataset.csv")
X_data = data.drop(["label"], axis=1)
y_data = data["label"]

X_data = X_data.values
y_data = y_data.values

# split into training and test datasets
x_trainval, x_test, y_trainval, y_test = train_test_split(X_data, y_data,
                                                          test_size=0.1,
                                                          stratify=y_data,
                                                          random_state=1004)

# normalize features
scaler = StandardScaler()
scaler.fit(x_trainval)
x_trainval = scaler.transform(x_trainval)
x_test = scaler.transform(x_test)

#############################################################################
# set params for gridesearch (tensorboard)
HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete([2, 3, 4]))
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([300, 600, 1000]))
HP_DROPOUT = hp.HParam('dropout_rate', hp.Discrete([0.2, 0.3]))

METRIC_ACCURACY = 'mean_accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning_results').as_default():
    hp.hparams_config(
        hparams=[HP_NUM_LAYERS, HP_NUM_UNITS, HP_DROPOUT],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Mean_accuracy')],
    )

# build a model
def train_model(x_train, y_train, hparams, log_dir, x_valid=None, y_valid=None, learning_rate=0.001, batchnorm=True, dropout=True, final=False):

    tf.keras.backend.clear_session()

    #----- model structure
    inputs = tf.keras.layers.Input(shape=(18,))
    x = inputs

    for i in range(hparams[HP_NUM_LAYERS]):
        x = tf.keras.layers.Dense(hparams[HP_NUM_UNITS])(x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        if dropout:
            x = tf.keras.layers.Dropout(hparams[HP_DROPOUT])(x)

    outputs = tf.keras.layers.Dense(10)(x)
    outputs = tf.keras.layers.Softmax()(outputs)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    #----- model compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    #----- model train
    ckpt_path = os.path.join(os.getcwd(), 'temp.h5')
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_weights_only=True,
        verbose=0
    )

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir)

    model.fit(x_train, y_train,
              validation_data=(x_valid, y_valid) if final is False else None,
              batch_size=1024,
              epochs=2000,
              callbacks=[ckpt, tensorboard_cb] if final is False else [tensorboard_cb],
              verbose=0)

    if not x_valid is None:
        model.load_weights(ckpt_path)

    return model

# build a stratifiedKFold process
def kfold_cv(x_trainval, y_trainval, hparams, run_dir, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits)
    accs = []
    j=1

    for train_index, valid_index in skf.split(x_trainval, y_trainval):
        print('CV Process: %d / %d ...' % (j, n_splits))
        x_train, x_valid = x_trainval[train_index], x_trainval[valid_index]
        y_train, y_valid = y_trainval[train_index], y_trainval[valid_index]

        trained_model = train_model(x_train, y_train, hparams,
                                    log_dir=os.path.join('logs/hparam_tuning/', run_dir, 'cv-%s' % j),
                                    x_valid=x_valid, y_valid=y_valid,
                                    )
        _, accuracy = trained_model.evaluate(x_valid, y_valid, verbose=0)
        accs.append(accuracy)
        with tf.summary.create_file_writer(os.path.join('logs/hparam_tuning/', run_dir, 'cv_summary')).as_default():
            tf.summary.scalar(METRIC_ACCURACY, accuracy, step=j)
        j += 1

    mean_accuracy = sum(accs) / len(accs)

    with tf.summary.create_file_writer(os.path.join('logs/hparam_tuning_results/', run_dir)).as_default():
        hp.hparams(hparams, trial_id=run_dir)
        tf.summary.scalar(METRIC_ACCURACY, mean_accuracy, step=1)

    return mean_accuracy

# grid search
s_results = {}

session_num = 0
start_time = datetime.now()
print("[%s] Start parameter search for the model" % start_time)

for num_layers in HP_NUM_LAYERS.domain.values:
    for num_units in HP_NUM_UNITS.domain.values:
        for dropout_rate in HP_DROPOUT.domain.values:
            hparams = {
                HP_NUM_LAYERS: num_layers,
                HP_NUM_UNITS: num_units,
                HP_DROPOUT: dropout_rate
                }
            run_name = "run-%d" % session_num

            print()
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})

            mean_accuracy = kfold_cv(x_trainval, y_trainval, hparams, run_name)

            s_results[run_name] = {'hparams': hparams,
                                   'mean_accuracy': mean_accuracy}
            session_num += 1

end_time = datetime.now()
duration_time = (end_time - start_time).seconds
print("[%s] Finish parameter search for the model (time: %d seconds)" % (end_time, duration_time))

# print best params
df_results = pd.DataFrame(s_results).T
best_param = df_results.sort_values(by='mean_accuracy', ascending=False).head(1)['hparams']
print('the best params:\n', {h.name: best_param.item()[h] for h in best_param.item()})

#############################################################################
# model retrain with whole trainval datasets and best params & model save
final_model = train_model(x_trainval, y_trainval, best_param.item(), log_dir='./logs/final_model', final=True)
final_model.save('./final_model.h5')

# load saved model and calcuate accuracy with test datasets
saved_model =tf.keras.models.load_model('./final_model.h5')
fin_accuracy_1 = accuracy_score(np.array(y_test), np.argmax(saved_model.predict(x_test), axis=-1))
print(fin_accuracy_1)