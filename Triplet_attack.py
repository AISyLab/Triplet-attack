import sys

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint

## for visualizing 
import matplotlib
import matplotlib.pyplot as plt, numpy as np
from sklearn.preprocessing import StandardScaler

import Util.SCA_dataset as datasets
import Util.Attack as Attack
import Util.DL_model as DL_model
import Util.Template_attack as TA
import Util.Triplet_loss as losses

if __name__ == "__main__":
    # File root for dataset and results
    data_root = 'Data/'
    model_root = 'Model/'
    result_root = 'Result/'

    # Dataset settings 
    # Note: Use code in Util/Generate_ASCAD_X.py to generate ASCAD datasets used in the paper (4,000 features, raw traces required)
    # Note: Attacking on the default ASCAD_F (700 features) or ASCAD_R (1,400 features) would work as well!
    datasetss = ['ASCAD','ASCAD_rand','AESHD'] # ['ASCAD','ASCAD_rand','AESHD']
    leakage_models = ['HW','ID'] # ['HW','ID']
    profiling_traces = 0 # 0: default
    noise_type = 'desync'
    noise_level = 0

    # Triplet settings
    classifier = 'Triplet'
    embedding_size = 32 # Dimension of output features
    alpha_values = 0 # 0: optimal
    margin = 0.4 # Triplet margin

    # Training settings
    batch_size = 512 # Triplet training batch size
    epochs = 1 # Triplet training epoch
    train_flag = True # True: train a model; False: load a model
    nb_attacks = 10 # number of attacks for GE calculation

    # Saving settings
    index = 0 # naming index
    save_folder = 'test'

    matplotlib.rcParams.update({'font.size': 12})

    # The data, split between train and test sets
    for dataset in datasetss:
        if dataset == 'ASCAD':
            correct_key = 224
            nb_traces_attacks = 10000
            (X_profiling, X_attack), (Y_profiling_ID,  Y_attack_ID), (plt_profiling,  plt_attack) = datasets.load_ascad(data_root+dataset+'/', profiling_traces=profiling_traces, leakage_model='ID')
        elif dataset == 'ASCAD_rand':
            correct_key = 34
            nb_traces_attacks = 10000
            (X_profiling, X_attack), (Y_profiling_ID,  Y_attack_ID), (plt_profiling,  plt_attack) = datasets.load_ascad_rand(data_root+dataset+'/', profiling_traces=profiling_traces, leakage_model='ID')
        elif dataset == 'AESHD':
            correct_key = 200
            nb_traces_attacks = 5000
            (X_profiling, X_attack), (Y_profiling_ID,  Y_attack_ID), (plt_profiling,  plt_attack) = datasets.load_aeshd(data_root+dataset+'/', profiling_traces=profiling_traces, leakage_model='ID')

        if noise_type=='gnoise':
            X_profiling = datasets.addGussianNoise(X_profiling, noise_level)
            X_attack = datasets.addGussianNoise(X_attack, noise_level)
        if noise_type=='desync':
            X_profiling = datasets.addDesync(X_profiling, int(noise_level))
            X_attack = datasets.addDesync(X_attack, int(noise_level))

        input_shape = (len(X_profiling[0]), 1)
        scaler = StandardScaler()
        X_profiling = scaler.fit_transform(X_profiling)
        X_attack = scaler.transform(X_attack)
        X_profiling = np.reshape(X_profiling, (len(X_profiling), X_profiling.shape[1], 1))
        X_attack = np.reshape(X_attack, (len(X_attack), X_attack.shape[1], 1))

        for leakage_model in leakage_models:
            test_info = '{}_{}_{}_ep{}_BS{}_ES{}_{}{}_alpha{}_margin{}_{}'.format(dataset, leakage_model, classifier, epochs, batch_size, embedding_size, noise_type, noise_level, alpha_values, margin, index)
            print('============={}============='.format(test_info))
            
            if leakage_model == 'HW':
                num_classes = 9
                Y_profiling = datasets.calculate_HW(Y_profiling_ID)
                Y_attack = datasets.calculate_HW(Y_attack_ID)
            else:
                num_classes = 256
                Y_profiling = Y_profiling_ID
                Y_attack = Y_attack_ID
            
            # set alpha value
            if alpha_values == 0:
                if dataset == 'AESHD':
                    alpha_value = 0.9
                else:
                    alpha_value = 0.1
            else:
                alpha_value = alpha_values

            # Network training...
            if train_flag:
                model = DL_model.pick(classifier, input_shape, embedding_size)
                model.summary()

                # train session
                opt = Adam(learning_rate=5e-4)
                loss = losses.triplet_semihard_loss(alpha_value, margin, num_classes)
                model.compile(loss=loss, optimizer=opt)

                filepath = model_root + 'Triplet_model_{}.hdf5'.format(test_info)
                checkpoint = ModelCheckpoint(
                    filepath=filepath,
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True)

                callbacks_list = [checkpoint]

                model.fit(
                    x=X_profiling,
                    y=Y_profiling,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_attack, Y_attack),
                    callbacks=callbacks_list,
                    verbose=2)
                
            model = load_model(model_root + 'Triplet_model_{}.hdf5'.format(test_info), compile=False)

            # Tamplate attack
            print('======Tamplate attack======')
            try:
                x_profiling_embeddings = model.predict(X_profiling)
                x_embeddings = model.predict(X_attack)
                mean_v,cov_v,classes = TA.template_training(x_profiling_embeddings,Y_profiling, pool=False)
                TA_prediction = TA.template_attacking_proba(mean_v,cov_v,x_embeddings,classes)
                avg_rank, all_rank = Attack.perform_attacks(nb_traces_attacks, TA_prediction, correct_key, plt_attack, log=True, dataset=dataset, nb_attacks=nb_attacks)
                print('GE: ', avg_rank[-1])
                print('GE smaller than 1: ', np.argmax(avg_rank < 1))
                print('GE smaller than 5: ', np.argmax(avg_rank < 5))
                print('Print and save GE TA')
                plt.plot(avg_rank)
                plt.xlabel('Number of Attack Traces')
                plt.ylabel('Guessing Entropy')
                # plt.savefig(result_root + save_folder + '/GE_{}.png'.format(test_info))
                # np.save(result_root + save_folder + '/GE_{}.npy'.format(test_info), avg_rank)
                plt.show()
                plt.clf()
            except np.linalg.LinAlgError as e:
                print(e)
                print('Tamplate attack error, singular matrix maybe')
