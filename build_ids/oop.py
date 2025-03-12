import os
import time
import seaborn as sns
import numpy as np
import pandas
#import tkinter as tk
from scipy import stats
from sklearn import utils
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import precision_score,recall_score,fbeta_score, confusion_matrix
import pandas as pd
import pickle, yaml
from sklearn.preprocessing import MinMaxScaler , FunctionTransformer
from sklearn.pipeline import make_pipeline
from skopt import BayesSearchCV
from skopt.plots import plot_objective
from skopt.space import Real, Integer, Categorical
from keras_tuner.tuners import BayesianOptimization
import tensorflow.keras as tf
from sklearn.metrics import roc_auc_score
from nfstream import NFStreamer


class Evaluation:
    def __init__(self):
        # I did not take any input from init but in each methods individually. Although I need to set args for each
        #methods, this way we wont force the initiated obj to be force to fulfill input args.
        # This clss is inherited by thy RF, so the obj from RF should not be force to fulfill input args.
        self.beta = 1
        self.pos_label = 1  # I put this here to use it different place which is also easier for change
    def precision_recall_f1_tn_fp_fn_tp_tpr_fpr_fdr_fnr_tnr(self, y_predict, y_true):
        p= self.precision(y_predict=y_predict,y_true=y_true)
        r = self.recall(y_predict=y_predict, y_true=y_true)
        f1 = self.fbeta(y_predict=y_predict, y_true=y_true)
        tn, fp, fn, tp = self.cofusion_matrix(y_predict=y_predict, y_true=y_true)
        fpr = self.fpr(fp, tn)
        tnr = self.tnr(tn,fp)
        tpr= self.tpr(tp,fn)
        fnr=self.fnr(fn,tp)
        fdr = self.fdr(fp,tp)
        return [p,r,f1,tn,fp,fn,tp,tpr,fpr,fdr,fnr,tnr]
    def tn_fp_fn_tp_fnr_tnr(self, y_predict, y_true):
        tn, fp, fn, tp = self.cofusion_matrix(y_predict=y_predict, y_true=y_true)
        fnr=self.fnr(fn,tp)
        tnr = self.tnr(tn, fp)
        return [tn, fp, fn, tp, fnr, tnr]
    def precision(self, y_predict, y_true):

        return precision_score(y_pred=y_predict, y_true=y_true, average='binary', pos_label=self.pos_label)
    def recall(self, y_predict, y_true):
        return recall_score(y_pred=y_predict, y_true=y_true, average='binary', pos_label=self.pos_label)
    def fbeta(self, y_predict, y_true, beta=1):
        return fbeta_score(y_pred=y_predict, y_true=y_true, average='binary', pos_label=self.pos_label, beta=beta)

    def cofusion_matrix(self, y_predict, y_true):
        tn, fp, fn, tp = confusion_matrix(labels=[0,1],y_true=y_true, y_pred=y_predict).ravel()
        return [tn, fp, fn, tp]
    def fnr(self, fn,tp):
        return fn/(tp+fn)
    def fpr(self, fp, tn):
        return fp /(fp+tn)
    def tpr(self, tp, fn):
        return tp /(tp+fn)
    def tnr(self, tn ,fp):
        return tn/(tn+fp)
    def fdr(self,fp,tp): # false discovery rate
        return fp/(tp+fp)
    def accuracy(self, tp,tn, fp, fn):
        return (tp+tn)/(tp+fp+fn+tn)
    def auc(self, y_predict_proba, y_true):# predict proba not hard decision
        return roc_auc_score(y_true=y_true, y_score=y_predict_proba)


class Plotter:
    def plot_histogram(self):
        print('plot_histogram')
    def plot_distro(self, column_list:list, colum_name_list:list, title:str,addr_2save=None):
        """
        :param column_list: a column should be pandas slice and send in a list
        :return:

        sns.histplot(data=a['bidirectional_mean_piat_ms'], label='Normal', kde=True, log_scale=True, stat="density",
                     linewidth=0, color='deepskyblue')
        sns.histplot(data=b['bidirectional_mean_piat_ms'], label='Abnormal', kde=True, log_scale=True, stat="density",
                     linewidth=0, color='tomato')
        """
        colors = ['red', 'blue','green', 'black']
        for i,col in enumerate(column_list):
            sns.kdeplot(col, label= colum_name_list[i], palette=[colors[i]],  fill=True)
        #plt.xlim(0, None)
        plt.legend(loc='upper right')
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.show()


    def plot_line(self):
        print('plot line')
    def plot_boxplot(self, data, labels:list, title:str, addr_2save=None):
        plt.boxplot(data,labels=labels)
        plt.ylabel('Score')
        plt.title(title)
        if addr_2save != None:
            plt.savefig(addr_2save)
        else:
            plt.show()
        plt.close()


    def plot_linechart(self):
        print('plot_linechart')
    def gan_plot_loss_accuracy(self, input_list:list, obj_name:str, plot_kind:str, addr_2save=None):
        measure = pd.DataFrame(input_list)
        measure.rename(columns={0:plot_kind}, inplace=True)
        plt.plot(measure)
        plt.title(obj_name + ' ' + plot_kind)
        plt.ylabel(plot_kind)
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper right')
        if addr_2save != None:
            plt.savefig(addr_2save)
        else:
            plt.show()
        plt.close()

    def gan_plot_accuracy(self, accuracy_list: list, obj_name: str, addr_2save=None):
        accuracy = pd.DataFrame(accuracy_list)
        accuracy.rename(columns={0: "loss"}, inplace=True)
        plt.plot(accuracy)
        plt.title(obj_name + ' accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper right')
        if addr_2save != None:
            plt.savefig(addr_2save)
        else:
            plt.show()
        plt.close()
    def plot_loss_validation(self, history,addr_2save=None):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        if addr_2save != None:
            plt.savefig(addr_2save)
        else:
            plt.show()
        plt.close()
    def plot_accuracy_validation(self):
        print('plot_accuracy_validation')
    def plot_roc_curve(self):
        print('plot_roc_curve')
class ids_util:
    def __init__(self):
        self.time_stamp = time.strftime("%d-%m-%Y-%H-%M-%S")

    def reshape_4cnn(self, x_train):
        return x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    def build_pipeline(self, ids_name:str):
        if ids_name in self.cnn_list:
            self.pipeline = make_pipeline(MinMaxScaler(), FunctionTransformer(self.reshape_4cnn))
            return self.pipeline
        else:
            self.pipeline = make_pipeline(MinMaxScaler())
            return self.pipeline
    def load_ids(self, ids_name: str, ids_addr: str, ds_addr_4pipeline: str , addr2save: str=None):
        # this takes ids name and addr and load it
        # however, we should also load training ds and make pipeline based on that
        # This way we gurantee the distribution of tarining and testing is identical
        self.ds_loader = Dataset()  # feature set is defined here
        self.eval = Evaluation()
        self.plotter = Plotter()
        self.cnn_list = ['CNN', 'cnn','nn', 'NN']
        self.rf_list = ['rf','RF', 'randomforest', 'rforest']
        self.knn_list = ['KNN','knn']
        # --------- create folder for loaded ds
        if addr2save != None:
            self.ids_folder_addr = addr2save + ids_name + '_' + self.time_stamp

            print('*+*+*+*+*+*+*+*+*+*+*+*+Create result folders for ' + ids_name.upper() + ' *+*+*+*+*+*+*+*+*+*+*+*+')
            print('Result folder address: {}'.format(addr2save))

            os.mkdir(self.ids_folder_addr)
            os.mkdir(self.ids_folder_addr + '/predict')
            os.mkdir(self.ids_folder_addr + '/plot')
            os.mkdir(self.ids_folder_addr + '/model')
        # --------load ds to make a pipeline
        ds = self.ds_loader.load_train_ds(ds_addr_4pipeline, print_info=False)
        # ----loading  IDS
        if ids_name in self.cnn_list:
            self.ids = tf.models.load_model(ids_addr)
            print('+++ Loaded Deployed IDS Model Summary +++\n', self.ids.summary())
            # after loading ids we also loads the ds that ids trained on
            # so we make a pipeline based on it and use self to globally declare self.ids_pipeline
            # This way we do not have problem in ids_predict method
            # So either we train a model or load, in both case we have the pipeline of the training
            self.ids_pipeline = self.build_pipeline(ids_name)
            self.ids_pipeline.fit_transform(ds.loc[:, ds.columns != 'label'])
            return [self.ids , self.ids_pipeline]
        else:
            with open(ids_addr, 'rb') as f:
                self.ids = pickle.load(f)
            print('+++ Loaded  IDS Model Summary +++\n', self.ids.__getstate__())
            self.ids_pipeline = self.build_pipeline(ids_name)
            self.ids_pipeline.fit_transform(ds.loc[:, ds.columns != 'label'])
            return [self.ids , self.ids_pipeline]

class Utils:
    def __init__(self):
        self.feature_set = ['bidirectional_packets','bidirectional_bytes','bidirectional_min_ps',
                            'bidirectional_mean_ps','bidirectional_stddev_ps', 'bidirectional_max_ps' ,
                            'bidirectional_min_piat_ms','bidirectional_mean_piat_ms',
                            'bidirectional_stddev_piat_ms', 'bidirectional_max_piat_ms', 'bidirectional_syn_packets' ,
                            'bidirectional_cwr_packets', 'bidirectional_ece_packets','bidirectional_urg_packets',
                            'bidirectional_ack_packets', 'bidirectional_psh_packets', 'bidirectional_rst_packets' ,
                            'bidirectional_fin_packets'
                            ,
                            'src2dst_packets', 'src2dst_bytes', 'src2dst_min_ps',
                            'src2dst_mean_ps',
                            'src2dst_stddev_ps', 'src2dst_max_ps', 'src2dst_min_piat_ms',
                            'src2dst_mean_piat_ms',
                            'src2dst_stddev_piat_ms', 'src2dst_max_piat_ms', 'src2dst_syn_packets',
                            'src2dst_cwr_packets', 'src2dst_ece_packets', 'src2dst_urg_packets',
                            'src2dst_ack_packets', 'src2dst_psh_packets', 'src2dst_rst_packets',
                            'src2dst_fin_packets'
                            ,
                            'dst2src_packets', 'dst2src_bytes', 'dst2src_min_ps',
                            'dst2src_mean_ps',
                            'dst2src_stddev_ps', 'dst2src_max_ps', 'dst2src_min_piat_ms',
                            'dst2src_mean_piat_ms',
                            'dst2src_stddev_piat_ms', 'dst2src_max_piat_ms', 'dst2src_syn_packets',
                            'dst2src_cwr_packets', 'dst2src_ece_packets', 'dst2src_urg_packets',
                            'dst2src_ack_packets', 'dst2src_psh_packets', 'dst2src_rst_packets',
                            'dst2src_fin_packets'     ]
    def load_yaml_config(self,addr:str):
        with open(addr, 'r') as f:
            self.config = yaml.safe_load(f)
        return  self.config
    def return_duplicates(self, ds, feature_list=None):
        if feature_list == None :
            df= ds.loc[:, ds.columns.isin([self.feature_set])]
        else:
            df = ds.loc[:, ds.columns.isin([feature_list])]
        return df.duplicated().sum()
    def return_null(self,ds, feature_list=None):
        if feature_list == None :
            df= ds.loc[:, ds.columns == self.feature_set]
        else:
            df = ds.loc[:, ds.columns == feature_list]
        return df.isnull().sum()
    def t_test(self, column1, column2):
        t_stat, p_value = ttest_ind(column1,column2)
        if p_value < 0.05:  # Choose an appropriate significance level
            print("There is a significant difference between the two distributions.")
        else:
            print("There is no significant difference between the two distributions.")
    def man_whitney_U_test (self, column1, column2):
        statistic, p_value = stats.mannwhitneyu(column1, column2)
        alpha = 0.05 #Set your significance level (alpha)
        if p_value < alpha:
            print("Reject the null hypothesis. There is a significant difference in packet sizes.")
        else:
            print("Fail to reject the null hypothesis. There is no significant difference in packet sizes.")
    def save_yaml_config(self, data, addr):
        print('save config')
        with open(addr, 'w') as f:
            yaml.dump(data, f)
    def pcap2csv(self, pcap_addr, csv2save_addr):
        global ds
        for i in range(1, 28):
            my_streamer = NFStreamer(
                source=pcap_addr,
                # Disable L7 dissection for readability purpose.
                n_dissections=0,
                statistical_analysis=True)
            ds = my_streamer.to_pandas()
        print("CSV File Shape: {}".format(ds.shape))
        ds.to_csv(csv2save_addr, index= False)
    def add_results_2csv(self):
        print('add result to csv')
    def load_results_fromCSV(self):
        print('load_results_fromCSV')
    def create_custome_dataFrame_2save_results(self):
        print('create_custome_dataFrame_2save_results')
    def add_results_2dataframe(self):
        print('add_results_2dataframe')
    def loadresult_fromDataFrame(self):
        print('loadresult_fromDataFrame')

class Dataset:
    def __init__(self):
        self.feature_set = ['bidirectional_packets','bidirectional_bytes','bidirectional_min_ps',
                            'bidirectional_mean_ps','bidirectional_stddev_ps', 'bidirectional_max_ps' ,
                            'bidirectional_min_piat_ms','bidirectional_mean_piat_ms',
                            'bidirectional_stddev_piat_ms', 'bidirectional_max_piat_ms', 'bidirectional_syn_packets' ,
                            'bidirectional_cwr_packets', 'bidirectional_ece_packets','bidirectional_urg_packets',
                            'bidirectional_ack_packets', 'bidirectional_psh_packets', 'bidirectional_rst_packets' ,
                            'bidirectional_fin_packets'
                            ,
                            'src2dst_packets', 'src2dst_bytes', 'src2dst_min_ps',
                            'src2dst_mean_ps',
                            'src2dst_stddev_ps', 'src2dst_max_ps', 'src2dst_min_piat_ms',
                            'src2dst_mean_piat_ms',
                            'src2dst_stddev_piat_ms', 'src2dst_max_piat_ms', 'src2dst_syn_packets',
                            'src2dst_cwr_packets', 'src2dst_ece_packets', 'src2dst_urg_packets',
                            'src2dst_ack_packets', 'src2dst_psh_packets', 'src2dst_rst_packets',
                            'src2dst_fin_packets'
                            ,
                            'dst2src_packets', 'dst2src_bytes', 'dst2src_min_ps',
                            'dst2src_mean_ps',
                            'dst2src_stddev_ps', 'dst2src_max_ps', 'dst2src_min_piat_ms',
                            'dst2src_mean_piat_ms',
                            'dst2src_stddev_piat_ms', 'dst2src_max_piat_ms', 'dst2src_syn_packets',
                            'dst2src_cwr_packets', 'dst2src_ece_packets', 'dst2src_urg_packets',
                            'dst2src_ack_packets', 'dst2src_psh_packets', 'dst2src_rst_packets',
                            'dst2src_fin_packets', 'label'
                            ]
        self.dataset_chunks = []
    def load_train_ds(self, addr, print_info = True):

        # when we use nfstream it gives us a list of features that we do not use all, so we pass the basic feature to use
        self.trainXY_ds = pd.read_csv(addr, usecols= self.feature_set)

        if print_info:
            if (self.trainXY_ds['label'].value_counts()[1] > 0):
                #print('================================================================')
                print(
                    'Train DS shape: {} ==> Benign:{} (ratio {:.2f}%) | Malicious: {}(ratio {:.2f}%) '.format(
                        self.trainXY_ds.shape, self.trainXY_ds['label'].value_counts()[0],
                        ((self.trainXY_ds['label'].value_counts()[0] / self.trainXY_ds.shape[0]) * 100),
                        self.trainXY_ds['label'].value_counts()[1],
                        ((self.trainXY_ds['label'].value_counts()[1] / self.trainXY_ds.shape[0]) * 100)))
            else:
                print(
                    'Test DS shape: {} ==> ONLY Benign:{} '.format(
                        self.trainXY_ds , self.trainXY_ds['label'].value_counts()[0]))
        return self.trainXY_ds
    def probe_slicing(self, total_probe_df, probe_range_list:list,  probe_2save_addr= '',probe_2save=False):
        # it used to create different proble size from a ds where benign and malicious of each are equal
        benign = total_probe_df[total_probe_df['label']== 0]
        malicious = total_probe_df[total_probe_df['label'] == 1]
        for rng in enumerate(probe_range_list):
            probeX = benign.sample(round(rng[1] / 2))
            probeY = malicious.sample(round(rng[1] / 2))
            probe_list=[probeX,probeY]
            probeXY= pd.DataFrame(pd.concat(probe_list, axis=0))
            if probe_2save:
                probeXY.to_csv(probe_2save_addr+'probeXY_'+str(rng[1])+'.csv',index=False)
    def print_info(self, ds):
        #print('================================================================')
        print(
            'DS shape: {} ==> Benign:{} (ratio {:.2f}%) | Malicious: {}(ratio {:.2f}%) '.format(
                ds.shape, ds['label'].value_counts()[0],
                ((ds['label'].value_counts()[0] / ds.shape[0]) * 100),
                ds['label'].value_counts()[1],
                ((ds['label'].value_counts()[1] / ds.shape[0]) * 100)))

    def label_ds_binary_class(self, df:pandas, victim_ip_list:list,c2_ip_list:list ,addr_2save:str):
        """

        :param df: input ds to add label column
        :param victim_ip_list:
        :param c2_ip_list:
        :param addr_2save: addr to save the new labeled ds
        :return:
        """
        df['label'] = 0 #put all 0
        df.loc[df["src_ip"].isin(victim_ip_list) & df["dst_ip"].isin(c2_ip_list), "label"] = 1
        df.loc[df["dst_ip"].isin(c2_ip_list) & df["src_ip"].isin(victim_ip_list), "label"] = 1
        pd.DataFrame(df).to_csv(addr_2save,  index=False)

    def add_ds_chunk_2list(self, ds):
        return self.ds_lists.append(ds)

    def combine_ds(self, ds_list:list, addr2save:str):
        """

        :param ds_list: full addr of indv csv file
        :param addr2save:
        :return:
        """
        ds = []
        for item in ds_list:
            df = pd.read_csv(item)
            ds.append(df)
        combined_ds =pd.concat(ds, axis=0)
        print(combined_ds.shape)
        combined_ds.to_csv(addr2save + 'red_team_probing_combined_ds.csv')
    def split_ds(self, ds_addr:str, add_2save, show_info:bool= False, n_split:int=5):
        ds_shuffeld = utils.shuffle(self.load_whole_ds(ds_addr))
        df_split = np.array_split(ds_shuffeld, n_split)
        test_list = [[0], [1], [2], [3], [4]]

        def to_combine_ds(a, b):
            combine = [a, b]
            return pd.DataFrame(pd.concat(combine, axis=0))
        def remove_keys(keylist, main_list):
            for item in keylist:
                # print (item)
                main_list.remove(item)
            return main_list
        for cont, item in enumerate(test_list):
            y = [0, 1, 2, 3, 4]
            train_list = remove_keys(item, y)
            print(train_list, item)
            train_combined = pd.concat(
                [df_split[(train_list[0])], df_split[(train_list[1])], df_split[(train_list[2])], df_split[(train_list[3])]])
            train_combined.to_csv("/media/sf_GAN_output/red-teaming-10-Sep-2023/4.comparison_with_CICIDSandUNSW/New DS/ds_combined/train" + str(cont) + ".csv")
            df_split[item[0]].to_csv(
                "/media/sf_GAN_output/red-teaming-10-Sep-2023/4.comparison_with_CICIDSandUNSW/New DS/ds_combined/test" + str(
                    cont) + ".csv")

    def load_test_ds(self,addr, print_info = True):
        self.testXY_ds = pd.read_csv(addr, usecols= self.feature_set)
        if print_info:
            if (len(self.testXY_ds['label'].value_counts()) > 1):
                #print('================================================================')
                print(
                    'Test DS shape: {} ==> Benign:{} (ratio {:.2f}%) | Malicious: {}(ratio {:.2f}%) '.format(
                        self.testXY_ds.shape, self.testXY_ds['label'].value_counts()[0],
                        ((self.testXY_ds['label'].value_counts()[0] / self.testXY_ds.shape[0]) * 100),
                        self.testXY_ds['label'].value_counts()[1],
                        ((self.testXY_ds['label'].value_counts()[1] / self.testXY_ds.shape[0]) * 100)))
            elif (len(self.testXY_ds['label'].value_counts()) <= 1):
                print(
                    'Test DS shape: {} ==> ONLY Benign:{} '.format(
                        self.testXY_ds.shape, self.testXY_ds['label'].value_counts()[0] ))
        return self.testXY_ds
    def load_whole_ds(self, addr,):
        print('loading the whole DS')
        self.whole_ds = pd.read_csv(addr, usecols= self.feature_set)
        return self.whole_ds
    def load_multiple_DS(self, addr_list):
        print ('loading multiple DSs')
        self.ds_lists = []
        for addr in addr_list:
            self.ds_lists.append(pd.read_csv(addr, usecols= self.feature_set))
        return self.ds_lists

    def save_ds(self, ds, addr):
        print('saving ds')
        pd.DataFrame(ds).to_csv(addr,  index=False)

class cnn_model:
     def __init__(self, input_shape = (54,1)):
         self.input_shape = input_shape
     def model_builder(self, hp):
        model = tf.Sequential()


        # -----activation
        hp_activation = hp.Choice('activation',
                                  values=['relu', 'elu', 'tanh'])  # same AF will applied for all layers

        # ---Conv layers:dropout after Conv is weird, see notes
        model.add(tf.layers.Conv1D(filters=32, kernel_size=5, input_shape=self.input_shape, activation=hp_activation,
                                   use_bias=False))  # X_train_shaped.shape[1]
        tf.layers.BatchNormalization()  # conv layer can also have normalization
        model.add(tf.layers.MaxPool1D(pool_size=2))

        model.add(tf.layers.Conv1D(filters=32, kernel_size=3, activation=hp_activation, use_bias=False))
        tf.layers.BatchNormalization()  # conv layer can also have normalization
        model.add(tf.layers.MaxPool1D(pool_size=2))

        # ---flatten
        model.add(tf.layers.Flatten())

        # --------learnign rate
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])  # 1e-3 in decimal 0.001

        # --------------Define the parameters ranges

        for i in range(hp.Int('dense_layers', 1, 5)):
            model.add(tf.layers.Dense(hp.Int('layer_' + str(i), min_value=100, max_value=512, step=32),
                                      activation=hp_activation, use_bias=False))
            tf.layers.BatchNormalization()
            tf.layers.Dropout(0.5)


        # ---Output layer
        model.add(tf.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(learning_rate=hp_learning_rate),
                      metrics=["accuracy"])
        return model

     # def tune(self, X_train_reshaped, label, tun_log_addr, max_trials = 10, tuner_callbad_patience=3,
     #          tuner_epoch= 15, model_plot_2save = False):  # help https://keras.io/guides/keras_tuner/getting_started/
     def tune(self, train_x, train_y,tuner_iteration, tun_log_addr,**kwargs):
        kwargs.setdefault('tuner_epoch', 15)
        kwargs.setdefault('model_plot_2save', True)
        kwargs.setdefault('tuner_callbad_patience', 3)
        tuner = BayesianOptimization(
            self.model_builder,
            objective='val_accuracy',
            max_trials=tuner_iteration , # the number of iteration with one architecture
            num_initial_points= int(.2 * tuner_iteration), # GPT: number of initial random points to sample in the search space before starting the optimization process. A reasonable starting point could be 10-20% of the total number of iterations (n_iter).
            executions_per_trial=1,
            directory='my_dir',
            overwrite=True,
            seed=48,
            project_name=tun_log_addr )

        #X_train_reshaped = train_set.reshape((train_set.shape[0], train_set.shape[1], 1))
        stop_early = tf.callbacks.EarlyStopping(monitor='val_loss', patience=kwargs['tuner_callbad_patience'])
        tuner.search(x=train_x, y=train_y, epochs=kwargs['tuner_epoch'], validation_split=0.2, callbacks=[stop_early],
                     verbose=0)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = tuner.hypermodel.build(best_hps)
        print('Best hyper parameters values: {}'.format(tuner.get_best_hyperparameters()[0].values))
        if kwargs['model_plot_2save']:
            tf.utils.plot_model(model,
                                show_layer_names=True, show_shapes=True,
                                to_file=tun_log_addr + 'best_model_plot.png')
        return [model, tuner.get_best_hyperparameters()[0].values]


class knn_model (KNeighborsClassifier):
    def __init__(self,n_neighbors=5,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        KNeighborsClassifier.__init__(self)
    def tune (self, train_x, train_y,cv,tuner_iteration= 10, **kwargs):
        kwargs.setdefault('plot', False)
        self.optimizer = BayesSearchCV(
            estimator=KNeighborsClassifier(),
            search_spaces={
                'n_neighbors': Integer(1, 21),
                'weights': Categorical(["uniform", "distance"]),
                'metric': Categorical(["euclidean", "manhattan", "minkowski"]),
            },
            cv=cv, # number of folds to use in the cross-validation process
            n_iter=tuner_iteration, # the number of iterations of the Bayesian optimization process
            n_jobs=-1,
            scoring='accuracy',
            verbose=1,
            random_state=42 # use similar seed to generate identical random numbers
        )
        self.optimizer.fit(train_x, train_y)
        print("val. score: %s" % self.optimizer.best_score_)
        print('best params: %s' % self.optimizer.best_params_)
        if kwargs['plot']:
            _ = plot_objective(self.optimizer.optimizer_results_[0],
                               dimensions=["n_neighbors", "weights", "metric"],
                               n_minimum_search=int(1e8))
            plt.show()
        return  self.optimizer.best_params_
class rf_model( RandomForestClassifier):
    def __init__(self, n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        RandomForestClassifier.__init__(self)

    def tune(self, train_x, train_y, tuner_iteration=10, **kwargs):
        kwargs.setdefault('cv', 2)
        kwargs.setdefault('plot', False)
        self.optimizer = BayesSearchCV(
            estimator=RandomForestClassifier(),
            search_spaces={
                'n_estimators': Integer(10, 300),
                'max_depth': Integer(2, 10),
                'min_samples_split': Integer(2, 10),
                'min_samples_leaf': Integer(1, 10),
                'max_features': Categorical(["sqrt", "log2"]),
            },
            cv=kwargs['cv'], # number of folds to use in the cross-validation process
            n_iter=tuner_iteration, # the number of iterations of the Bayesian optimization process
            n_jobs=-1,
            scoring='accuracy',
            verbose=1,
            random_state=42 # use similar seed to generate identical random numbers
        )
        self.optimizer.fit(train_x, train_y)
        print("val. score: %s" % self.optimizer.best_score_)
        print('best params: %s' % self.optimizer.best_params_)
        if kwargs['plot']:
            _ = plot_objective(self.optimizer.optimizer_results_[0],
                               dimensions=["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"],
                               n_minimum_search=int(1e8))
            plt.show()
        return  self.optimizer.best_params_
        #print("test score: %s" % self.optimizer.score(X_test, y_test))
        #super().__init__()
        """
        RandomForestClassifier.__init__(self): This directly calls the constructor of the RandomForestClassifier class 
        and initializes the object. It is an explicit way of initializing the base class and can be useful in certain 
        scenarios where you want to specifically call the base class constructor with custom arguments.

        super().__init__: This is a more flexible and recommended approach for initializing the base class. 
        It allows you to call the constructor of the immediate base class without explicitly mentioning the class name.
         It automatically determines the base class based on the inheritance hierarchy. By using super().__init__, you 
         ensure that the initialization process goes through all the necessary steps defined by the inheritance chain.

        In most cases, it is preferred to use super().__init__ because it provides a more maintainable and flexible way 
        of initializing the base class, especially in scenarios where you have multiple levels of inheritance. 
        It allows you to easily update the inheritance hierarchy without modifying every constructor call.
        """

def main():
    """
    pipeline = make_pipeline(MinMaxScaler())
    rf= rf_model()

    train = rf.load_train_ds('/home/mehrdad/PycharmProjects/GAN-Framework/dataset/CICIDS/probe/probeXY_1000.csv')
    #y_train = pd.read_csv('/home/mehrdad/PycharmProjects/GAN-Framework/transferability/model_extraction_results/results_10-01-2023-18-27-00/subs_model/target-rf_unsw_probe_unsw_100_subs-rf/labeled_probe_class_prediction')
    param = rf.tune(5,32)
    rf.set_params(**param)
    y_train = train.loc[:, train.columns == 'label'].values.ravel('F')
    x_train_transformed = pipeline.fit_transform(train.loc[:, train.columns != 'label'])

    test = rf.load_test_ds('/home/mehrdad/PycharmProjects/GAN-Framework/dataset/UNSW/split/nested-kfold/deployment_test/deployment_test.csv')

    y_test = test.loc[:, test.columns == 'label'].values.ravel('F')
    x_test_transformed = pipeline.transform(test.loc[:, test.columns != 'label'])

    rf.fit(x_train_transformed,y_train)
    xyhat = rf.predict(x_test_transformed)
    print (rf.fbeta(y_predict=xyhat, y_true=y_test))"""
    ds = Dataset()
    ds.probe_slicing(pd.read_csv('/media/sf_GAN_output/red-teaming-10-Sep-2023/4.comparison_with_CICIDSandUNSW/New DS/red_team_probing_combined_ds.csv'),
                     [100,200,300,400,500,600,700,800,900,1000],
                     '/media/sf_GAN_output/red-teaming-10-Sep-2023/4.comparison_with_CICIDSandUNSW/New DS/probe_slice/',
                     probe_2save=True)
    """df = pd.read_csv("/media/sf_GAN_output/red-teaming-10-Sep-2023/GAN- Red-team Project/pcap/csv/labeled_sc-01-beacon_RCE_exfil_5mb.csv")#.values
    df2 = pd.read_csv("/media/sf_GAN_output/red-teaming-10-Sep-2023/GAN- Red-team Project/pcap/csv/labeled_sc-02-randomize-beacon_RCE_exfil_5mb.csv")
    df3 = pd.read_csv("/media/sf_GAN_output/red-teaming-10-Sep-2023/4.comparison_with_CICIDSandUNSW/New DS/labeled_sc-8_filtered_Normal_exploit_win_ICMP-exfil_SMTP.csv")
    col = "bidirectional_mean_ps"
    plot_list = []
    plot_list.append(df[df['label'] == 1].loc[:, df.columns == col].values)
    plot_list.append(df2[df2['label'] == 1].loc[:, df2.columns == col].values)
    plot_list.append( df3[df3['label'] == 0].loc[:, df3.columns == col])
    plot_list.append(df3[df3['label'] == 1].loc[:, df3.columns == col])"""
    #df= pd.read_csv( "/media/sf_GAN_output/red-teaming-10-Sep-2023/GAN- Red-team Project/pcap/csv/sc_03_beacon4prob.csv")
    #ds.label_ds_binary_class(df, [ "10.11.54.81","10.11.54.219","10.11.54.254",
     #                             "10.11.54.14","10.11.54.191","10.11.54.179", "10.11.54.142"],
      #                       ["10.11.54.37"],"/media/sf_GAN_output/red-teaming-10-Sep-2023/GAN- Red-team Project/pcap/csv/labeled_sc_03_beacon4prob.csv")
    util = Utils()
    #util.pcap2csv("/media/sf_GAN_output/red-teaming-10-Sep-2023/GAN- Red-team Project/pcap/sc_03_beacon4prob.pcap", "/media/sf_GAN_output/red-teaming-10-Sep-2023/GAN- Red-team Project/pcap/csv/sc_03_beacon4prob.csv")
    pl = Plotter()
    """util = Utils()
    util.t_test(plot_list[0],plot_list[1])
    util.man_whitney_U_test(plot_list[0], plot_list[1])"""
    #col_name = ['Beaconing', 'Random Beaconing','Benign' , 'Exfil_over_SMTP']
    #pl.plot_distro(plot_list, colum_name_list=col_name, title=col)
if __name__ == "__main__":
    #pass
    main()

