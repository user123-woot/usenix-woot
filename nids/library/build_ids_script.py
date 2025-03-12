import random
import sys 
sys.path.append("/C2_communication/nids/library")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler ,FunctionTransformer
from sklearn.ensemble import  RandomForestClassifier
from oop import rf_model, Dataset,Evaluation, Plotter, Utils, knn_model
import pandas as pd
from oop import cnn_model
from oop import cnn_model
import os, time, pickle, json
import glob, gc
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
import tensorflow.keras as tf
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# This setting effectively disables TensorFlow's logging messages, including INFO and WARNING messages, and only displays ERROR messages.

class build_ids:
    def __init__(self, config_addr= None):
        #The ids class is based on cross model concept where we have on ds but various models to be trained on this ids
        # build ids without ds makes no scence, so config_addr it mandatory parametrs
        #Even if we wanna load previous trained ids we need this
        if config_addr is not None:
            self.config = Utils().load_yaml_config(config_addr)
        self.ds_loader = Dataset()  # feature set is defined here
        self.eval = Evaluation()
        self.plotter = Plotter()
        self.results= pd.DataFrame({'model_name':[],'model_param':[],'precision':[],'recall':[],'f1_score':[],
                       'tn':[], 'fp':[], 'fn':[], 'tp':[],'tpr':[],'fpr':[],'fdr':[],'fnr':[], 'tnr':[], 'auroc'
                       'train_time':[],'test_time':[], 'description':[]})
        self.time_stamp = time.strftime("%d-%m-%Y-%H-%M-%S")

        #---make list for model factory if statment
        self.cnn_list = ['CNN', 'cnn','nn', 'NN']
        self.rf_list = ['rf','RF', 'randomforest', 'rforest']
        self.knn_list = ['KNN','knn']
        self.train_time = ''
        self.test_time = ''
    def build_pipeline(self, ids_name:str):
        if ids_name in self.cnn_list:
            #self.pipeline = make_pipeline(MinMaxScaler(), PCA(n_components=0.99),FunctionTransformer(self.reshape_4cnn))
            self.pipeline = make_pipeline(MinMaxScaler(),FunctionTransformer(self.reshape_4cnn))

            return self.pipeline
        else:
            self.pipeline = make_pipeline(MinMaxScaler())
            return self.pipeline

    def reshape_4cnn(self, x_train):
       return x_train.reshape((x_train.shape[0], x_train.shape[1], 1))


    def add2results(self, addr,model_name, train_ds_name, precision, recall, f1,
                    tn,fp, fn, tp, tpr, fpr, fdr, fnr, tnr,auroc, train_time, test_time, description, feature_set):
       new_row = {'model_name': [(model_name)], 'train_ds_name':[train_ds_name],
                  'precision': [precision], 'recall': [recall], 'f1_score': [f1],
                  'tn': [tn], 'fp': [fp], 'fn': [fn], 'tp': [tp], 'tpr': [tpr], 'fpr': [fpr],
                  'fdr': [fdr], 'fnr': [fnr], 'tnr': [tnr], 'auroc':[auroc],
                  'train_time': [train_time],
                  'test_time': [test_time],'description': [description], 'feature_set':[feature_set]}
       if os.path.exists(addr):
        pd.DataFrame(new_row).to_csv(addr, mode= 'a',index=False, header = False)
       elif not os.path.exists(addr):
           pd.DataFrame(new_row).to_csv(addr, mode='a', index=False)

    def return_best_param_from_result_csv(self, csv_addr):
        # this apply majority voting to choose the best param with highest f1 in 5 iterations
        #so it loads the result report csv and return a dic of param
        # this param dic can be use to build deployment ids and build the model with it
        final = pd.read_csv(csv_addr)
        idx = final[final['model_param'].isin(final['model_param'].mode().tolist())]['f1_score'].idxmax()
        json_accepted_string = (final[final.index == idx]['model_param'][idx]).replace ("'", "\"") # load param str and make it json-ready
        best_params_4deployment = json.loads(json_accepted_string) #convert str to dic
        return [best_params_4deployment , idx]
    def load_ids(self, ids_name:str , ids_addr:str , ds_addr_4pipeline:str, addr2save:str= None ):
        # this takes ids name and addr and load it
        # however, we should also load training ds and make pipeline based on that
        # This way we gurantee the distribution of tarining and testing is identical

        #--------- create folder for loaded ds
        if addr2save != None:
            # ORIGINAl ==> self.ids_folder_addr = addr2save +'_' +ids_name + '_' + self.time_stamp
            self.ids_folder_addr = addr2save +'_' +ids_name 
            print('*+*+*+*+*+*+*+*+*+*+*+*+Create result folders for ' + ids_name.upper() + ' *+*+*+*+*+*+*+*+*+*+*+*+')
            print('Result folder address: {}'.format(addr2save))

            os.makedirs(self.ids_folder_addr , exist_ok=True )
            os.makedirs(self.ids_folder_addr+ '/predict' , exist_ok=True )
            os.makedirs(self.ids_folder_addr + '/predict', exist_ok=True )
            os.makedirs(self.ids_folder_addr+ '/predict' , exist_ok=True )

        #--------load ds to make a pipeline
        ds = self.ds_loader.load_train_ds(ds_addr_4pipeline, print_info=False)
        #----loading  IDS
        if ids_name in self.cnn_list:
            self.ids= tf.models.load_model(ids_addr)
            print('+++ Loaded Deployed IDS Model Summary +++\n',self.ids.summary())
            # after loading ids we also loads the ds that ids trained on
            # so we make a pipeline based on it and use self to globally declare self.ids_pipeline
            # This way we do not have problem in ids_predict method
            # So either we train a model or load, in both case we have the pipeline of the training
            self.ids_pipeline = self.build_pipeline(ids_name)
            self.ids_pipeline.fit_transform(ds.loc[:, ds.columns != 'label'])
            return self.ids
        else:
            with open(ids_addr , 'rb') as f:
                self.ids = pickle.load(f)
            #print('+++ Loaded  IDS Model Summary +++\n', self.ids.__getstate__())
            self.ids_pipeline = self.build_pipeline(ids_name)
            self.ids_pipeline.fit_transform(ds.loc[:, ds.columns != 'label'])
            return self.ids
    def build_new_ids_from_loaded_ids(self, ids_name:str, ids_addr: str):
        # it loads a stored ids, and build a new ids based on loaded ids parameters
        global new_ids
        if ids_name in ["CNN", "cnn", "nn"]:
            model = tf.models.load_model(ids_addr)  # load the model
            lr = model.optimizer.get_config()['learning_rate']  # get_config returns all opt params
            cloned_model = tf.models.clone_model(model)  # cloning
            cloned_model.compile(loss=tf.losses.get(model.loss).__name__,
                                 optimizer=tf.optimizers.Adam(learning_rate=lr),
                                 metrics=[model.metrics[1].name])
            return cloned_model
        else:
            with open(ids_addr, 'rb') as f:
                ids = pickle.load(f)
            params = ids.get_params(deep=True)
            if ids_name in ["rf", "RF", "rf_model"]:
                 new_ids = RandomForestClassifier()
            elif ids_name in ["knn", "KNN", "knn_model"]:
                new_ids = KNeighborsClassifier()
            new_ids.set_params(**params)
            return new_ids
    def ids_obj_creator(self, ids_name, input_train_shape =None):
        #This model factory pattern for making  obj
        # if want to add mode clf we should make the class in oop_code.py and then extend it here
        if ids_name in self.cnn_list:
            if input_train_shape is not None:
                ids = cnn_model(input_shape=(input_train_shape, 1))
                return ids
            else:
                print('input shape for NN model is missing')
                exit()
        elif ids_name in self.rf_list:
            ids = rf_model()
            return ids
        elif ids_name in self.knn_list:
            ids = knn_model()
            return ids
        else: print('Error: IDS name {} not found'.format(ids_name))

    def ids_train(self, ids,ids_name:str ,train_x, y_train, addr2save:str, save_model:bool):
        # train the input ids which is build somewhere else

        self.ids = ids
        self.ids_folder_addr = addr2save + ids_name + '_' + self.time_stamp

        print('*+*+*+*+*+*+*+*+*+*+*+*+Create result folders for ' + ids_name.upper() + ' *+*+*+*+*+*+*+*+*+*+*+*+')
        print('Result folder address: {}'.format(addr2save))

        os.mkdir(self.ids_folder_addr)
        os.mkdir(self.ids_folder_addr + '/predict')
        os.mkdir(self.ids_folder_addr + '/plot')
        os.mkdir(self.ids_folder_addr + '/model')
        self.ids_pipeline = self.build_pipeline(ids_name)
        x_train_transformed = self.ids_pipeline.fit_transform(train_x.loc[:, train_x.columns != 'label'])
        print('-----------------Start Training ---------------')
        if ids_name in self.cnn_list:
            stop_early = tf.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            start_train = time.time()
            history = self.ids.fit(x_train_transformed, y_train, epochs=30, batch_size=256,validation_split=0.2, verbose=1, shuffle=False, callbacks=[stop_early])
            stop_train = time.time()
            self.plotter.plot_loss_validation(history, self.ids_folder_addr +'/plot/val_loss.png')
        else:
            start_train = time.time()
            self.ids.fit(x_train_transformed, y_train)
            stop_train = time.time()
        self.train_time = stop_train-start_train
        print('--------------- Train Time: {} -----------------'.format(stop_train - start_train))
        if save_model:
            # save model
            if ids_name in self.cnn_list:
                self.ids.save(self.ids_folder_addr + '/model/' + '' + ids_name + '_model.keras')
            else:
                pickle.dump(self.ids, open(self.ids_folder_addr + '/model/' + ids_name + '_model', 'wb'))



    def ids_build_tune_train(self, ids_name:str,train_ds_name:str,train_x, y_train, addr2save:str, model2save:bool):
        """
        def check_keywords(text, keywords): 
            found_keywords = [keyword for keyword in keywords if keyword in text] 
            return found_keywords
        train_ds_name= check_keywords(train_ds_name,["athen","freyja","empire"])
        """
        self.ids_folder_addr = f"{addr2save}Building_{ids_name}'_'on_{train_ds_name}_{self.time_stamp}"
        #self.ids_folder_addr = addr2save + ids_name + f"_trained_on_{train_ds_name}"
    
        print('*+*+*+*+*+*+*+*+*+*+*+*+Create result folders for ' + ids_name.upper() + ' *+*+*+*+*+*+*+*+*+*+*+*+')
        print('Result folder address: {}'.format(addr2save))

        os.makedirs(self.ids_folder_addr , exist_ok=True )
        os.makedirs(self.ids_folder_addr+ '/predict' , exist_ok=True )
        os.makedirs(self.ids_folder_addr + '/plot', exist_ok=True )
        os.makedirs(self.ids_folder_addr + '/model', exist_ok=True )

        #os.makedirs(self.ids_folder_addr+ f"{ids_name}_trained_{train_ds_name}" , exist_ok=True )

        print('*+*+*+*+*+*+*+*+*+*+*+*+Start Building '+ids_name.upper()+ ' *+*+*+*+*+*+*+*+*+*+*+*+')
        self.ids_pipeline = self.build_pipeline(ids_name)
        x_train_transformed = self.ids_pipeline.fit_transform(train_x) # train_x is already without label in input

        #-------build and tuning
        print('-------Start Building and Tuning '+ids_name.upper()+f"Training size after PCA: {x_train_transformed.shape}" +'-------')
        if ids_name in self.cnn_list:
            ids = self.ids_obj_creator(ids_name, input_train_shape=x_train_transformed.shape[1])
            self.ids,  _best_params = ids.tune(train_x=x_train_transformed, train_y=y_train, cv=2,tuner_iteration= 3,tun_log_addr =self.ids_folder_addr +'/model/' ,model_plot_2save=True)
        else:
            ids = self.ids_obj_creator(ids_name)
            # take 10% of input for tuning because the training samples was 500k
            subsample_size = int(len(x_train_transformed) * .2) #take 20% for validation
            subsmaple_index = np.random.choice(x_train_transformed.shape[0],size=subsample_size, replace=False )
            _best_params = ids.tune(train_x=x_train_transformed[subsmaple_index, :], train_y=y_train[subsmaple_index],cv=2)
            self.ids = ids.set_params(**_best_params)
            self.ids.set_params(**{'n_jobs':-1})
        #----save hyper parameters
        print('--------------- Store hyperparameters --------------- ')
        with open(self.ids_folder_addr + '/model/' + 'best_hps_iteration_' + '.json', 'w') as f:
            json.dump(_best_params, f)

        # ----: train
        if ids_name in self.cnn_list:
            stop_early = tf.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            start_train = time.time()
            history = self.ids.fit(x_train_transformed, y_train, epochs=30, batch_size=256,validation_split=0.2, verbose=1, shuffle=False, callbacks=[stop_early])
            stop_train = time.time()
            self.plotter.plot_loss_validation(history, self.ids_folder_addr +f"/plot/val_loss_{ids_name}_trained_{train_ds_name}.png")
        else:
            start_train = time.time()
            self.ids.fit(x_train_transformed, y_train)#.values.ravel())
            stop_train = time.time()
        print('--------------- Train Time: {} -----------------'.format(stop_train-start_train))
        self.train_time= stop_train-start_train

        if model2save:
            if ids_name in self.cnn_list:
                self.ids.save(self.ids_folder_addr + '/model/' + ids_name + f"_trained_on_{train_ds_name}"+'_model.keras', 'wb')
            else:
                print(f"Storing the trained model: {ids_name}")
                pickle.dump(self.ids, open(self.ids_folder_addr + '/model/' + ids_name +  f"_trained_on_{train_ds_name}"+'_model', 'wb'))
    def ids_predict(self, ids_name:str, test_x,test_y, save_predict:bool, eval2save:bool, save_model:bool,test_ds_name:str = '--', train_x_name:str="", feature_group=None):
        """
        if feature_group== 0:
            feature_set = "Zero adversarial feature in train"
        elif feature_group == 1:
            feature_set = "Top 50 percent feature is included in train"
        elif feature_group == None:
            feature_set = "Consider all 54 features to be pass through PCA"
        """
        # -----: scale test
        global stop_test, start_test, xyhat_proba, xyhat
        if not (hasattr(self, 'ids_pipeline') and self.ids_pipeline is not None):
            print('Error: training pipeline is not loaded for transforming test set')
            exit()
        else:
            x_test_transformed = self.ids_pipeline.transform(test_x.loc[:, test_x.columns != 'label'])

            # -----predict on test ds
            if ids_name in self.cnn_list:
                start_test = time.time()
                xyhat_proba = self.ids.predict(x_test_transformed)
                stop_test = time.time()
                xyhat = (xyhat_proba > 0.5).astype("int32")
            else:
                start_test = time.time()
                xyhat = self.ids.predict(x_test_transformed)
                stop_test = time.time()
                xyhat_proba = self.ids.predict_proba(x_test_transformed)
        #print('--------------- Test Time: {} -----------------'.format(stop_test-start_test))
        self.test_time = stop_test-start_test
        #"""
        #--------save prediction results
        if save_predict:
            pd.DataFrame(xyhat).to_csv(self.ids_folder_addr+'/predict/'+ids_name+f"_trained_on_{train_x_name}_predict_on_{test_ds_name}"+'_xyhat.csv',
                                          index=False)
            pd.DataFrame(xyhat_proba).to_csv(self.ids_folder_addr +'/predict/'+ids_name+f"_trained_on_{train_x_name}_predict_on_{test_ds_name}"+'_xyhat_proba.csv', index=False)

        #----------performance evaluation
        precision, recall, f1, tn, fp, fn, tp, tpr, fpr, fdr, fnr, tnr = self.eval.precision_recall_f1_tn_fp_fn_tp_tpr_fpr_fdr_fnr_tnr(
                xyhat, test_y)
        print( "precision: {} recall: {} f1: {} tn:{} fp: {} fn: {} tp: {} tpr: {} fpr: {} fdr: {} fnr: {} tnr: {}".format(
                precision, recall, f1, tn, fp, fn, tp, tpr, fpr, fdr, fnr, tnr))
        if ids_name in self.cnn_list:
                auroc = roc_auc_score(y_score=xyhat_proba, y_true=test_y)
        else:
            auroc = roc_auc_score(y_score=xyhat_proba[:, 1], y_true=test_y)
        print(f"AUROC:{auroc}")
        # ----- save results
        if eval2save:
            report2save ="/GAN-Framework/transferability/all_c2_results_advr_feature_exclusion2/"
            self.add2results(report2save+'/all_excluding_advr_feature_results.csv', model_name=ids_name ,train_ds_name=train_x_name, precision= precision,recall= recall,
            f1=f1, tn=tn, fp=fp, fn=fn, tp=tp, tpr=tpr, fpr=fpr, fdr=fdr,fnr= fnr, tnr=tnr,auroc=auroc,train_time=self.train_time, test_time=self.test_time ,
            description = test_ds_name, feature_set = f"excluded feature from target:{feature_group}")
        # save model
        if save_model:
            if ids_name in self.cnn_list:
                self.ids.save(self.ids_folder_addr +'/model/' + ''+ids_name+f"_trained_on_{train_x_name}"+'_model.keras')
            else:
                pickle.dump(self.ids, open(self.ids_folder_addr +'/model/'+ids_name+f"_trained_on_{train_x_name}"+'_model', 'wb'))
        #"""
        return [xyhat, xyhat_proba]

    #def ids_train_evaluate_kfold(self, ids_name:str,  eval2save:bool):
        #--------create  result folder
        if self.config["dataset_name"] in ["cicids2017", "cicids", "CICIDS2017", "CIC", "cic", "CICIDS"]:
            os.mkdir(
                self.config["ids_folder"]+'Building'+ids_name+'_IDS_on_CICIDS2017_results' + '_' + self.time_stamp)  # to save all info in this folder
            self.result_folder_address = self.config["ids_folder"]+'Building'+ids_name+'_IDS_on_CICIDS2017_results'  + '_' + self.time_stamp + '/'
        elif self.config["dataset_name"] in ["unsw", "UNSW"]:
            os.mkdir(self.config["ids_folder"]+ 'Building'+ids_name+'_IDS_on_unsw_results' + '_' + self.time_stamp)  # to save all info in this folder
            self.result_folder_address = self.config["ids_folder"]+'Building'+ids_name+'_IDS_on_unsw_results'+ '_' + self.time_stamp + '/'

        elif self.config["dataset_name"] in ["red","red-team", "red_team", "redteam"]:
            os.mkdir(self.config["ids_folder"]+ 'Building_'+ids_name+'_IDS_on_red_team_results' + '_' + self.time_stamp)  # to save all info in this folder
            self.result_folder_address = self.config["ids_folder"]+'Building_'+ids_name+'_IDS_on_red_team_results'+ '_' + self.time_stamp + '/'
        else:
            print('dataset name not found')
            exit()
        for i in range(len(self.config["train_ds_address"])):
            print('----------- Iteration {} ------------'.format(i))
            # -----folder creation to save results
            print('*+*+*+*+*+*+*+*+*+*+*+*+Create result folders for '+ids_name.upper()+ ' *+*+*+*+*+*+*+*+*+*+*+*+')
            print('Result folder address: {}'.format(self.result_folder_address ))

            os.mkdir(self.result_folder_address + 'fold' + str(i))
            os.mkdir(self.result_folder_address + 'fold' + str(i) + '/'+ ids_name)
            os.mkdir(self.result_folder_address + 'fold' + str(i) + '/'+ids_name+'/predict')
            os.mkdir(self.result_folder_address + 'fold' + str(i) + '/'+ids_name+'/plot')
            os.mkdir(self.result_folder_address + 'fold' + str(i) + '/'+ids_name+'/model')
            # -------load DS
            print('-------Loading ' + self.config["dataset_name"] + ' DS-------')
            train_x = (self.ds_loader.load_train_ds(addr=self.config["train_ds_address"]["fold" + str(i)]))#.sample(100)
            y_train = (train_x.loc[:, train_x.columns == 'label'].values.ravel('F'))
            test = (self.ds_loader.load_test_ds(addr=self.config["test_ds_address"]["fold" + str(i)]))#.sample(100)
            y_test = (test.loc[:, test.columns == 'label'].values.ravel('F'))

            print('*+*+*+*+*+*+*+*+*+*+*+*+Start Building '+ids_name.upper()+ ' *+*+*+*+*+*+*+*+*+*+*+*+')
            ids_pipeline = self.build_pipeline(ids_name)
            x_train_transformed = ids_pipeline.fit_transform(train_x.loc[:, train_x.columns != 'label'])

            # ------: build model with best parameters
            ids = self.ids_obj_creator(ids_name, input_train_shape=x_train_transformed.shape[1])

            #---start tuning
            print('-------Start Tuning '+ids_name.upper()+ '-------')
            if ids_name in self.cnn_list:
                ids,  _best_params = ids.tune(train_x= x_train_transformed,train_y=y_train, model_plot_2save=True, tuner_iteration=5,tun_log_addr=self.result_folder_address + 'fold' + str(i) + '/'+ids_name+'/model/')
                with open(self.result_folder_address + 'fold' + str(i) + '/'+ids_name+'/model/' + 'best_hps_iteration_' + str(i) + '.json','w') as f:
                    json.dump( _best_params, f)
            else:
                # take 10% of input for tuning because the training samples was 500k
                subsample_size = int(len(x_train_transformed)*.1)
                subsmaple_index = random.sample(range(x_train_transformed.shape[0]),subsample_size)
                _best_params = ids.tune(train_x=x_train_transformed[subsmaple_index, :], train_y=y_train[subsmaple_index], cv=2)
                ids.set_params(**_best_params)
                ids.set_params(**{'n_jobs':-1})

            # ----: train
            print('-------Start Training ' + ids_name.upper() + f"Training shape: {x_train_transformed.shape}"+'-------')
            if ids_name in self.cnn_list:
                stop_early = tf.callbacks.EarlyStopping(monitor='val_loss', patience=5)
                start_train = time.time()
                history = ids.fit(x_train_transformed, y_train, epochs=30, batch_size=256,validation_split=0.2, verbose=1, shuffle=False, callbacks=[stop_early])
                stop_train = time.time()
                self.plotter.plot_loss_validation(history,
                            self.result_folder_address + 'fold' + str(i) + '/'+ids_name+f"/plot/val_loss.png")
            else:
                start_train = time.time()
                ids.fit(x_train_transformed, y_train)
                stop_train = time.time()

            # -----: scale test
            x_test_transformed = ids_pipeline.transform(test.loc[:, test.columns != 'label'])

            # -----evaluate & save
            if ids_name in self.cnn_list:
                start_test = time.time()
                xyhat = ids.predict_classes(x_test_transformed)
                stop_test = time.time()
                xyhat_proba = ids.predict(x_test_transformed)
            else:
                start_test = time.time()
                xyhat = ids.predict(x_test_transformed)
                stop_test = time.time()
                xyhat_proba = ids.predict_proba(x_test_transformed)

            #--------save prediction results

            pd.DataFrame(xyhat).to_csv(self.result_folder_address + 'fold' + str(i) + '/'+ids_name+'/predict/'+ids_name+'_xyhat.csv',
                                          index=False)
            pd.DataFrame(xyhat_proba).to_csv(self.result_folder_address + 'fold' + str(i) + '/'+ids_name+'/predict/'+ids_name+'_xyhat_proba.csv', index=False)
            precision, recall, f1, tn, fp, fn, tp, tpr, fpr, fdr, fnr, tnr = self.eval.precision_recall_f1_tn_fp_fn_tp_tpr_fpr_fdr_fnr_tnr(
                xyhat, y_test)
            print( "precision: {} recall: {} f1: {} tn:{} fp: {} fn: {} tp: {} tpr: {} fpr: {} fdr: {} fnr: {} tnr: {}".format(
                    precision, recall, f1, tn, fp, fn, tp, tpr, fpr, fdr, fnr, tnr))


            # ----- save results
            if eval2save:
                self.add2results(self.result_folder_address + ids_name +'_results_report.csv', ids_name, str(_best_params), i, precision, recall,
                        f1, tn, fp, fn, tp, tpr, fpr, fdr, fnr, tnr, stop_train - start_train, stop_test - start_test,
                        ids_name+' prediction on test data for iteration: '
                        + str(i))

            # save model
            if ids_name in self.cnn_list:
                ids.save(self.result_folder_address + 'fold' + str(i) + '/'+ids_name+'/model/' + ''+ids_name+'_model')
            else:
                pickle.dump(ids, open(self.result_folder_address + 'fold' + str(i) + '/'+ids_name+'/model/'+ids_name+'_model', 'wb'))


def file_address_resolver (folder_address:str, extention:str):
    return glob.glob (folder_address+ f"/*.{extention}")
def list_csv_files(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files
