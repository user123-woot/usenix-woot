import pickle
import time
from oop import rf_model, Dataset, Evaluation, Utils
from sklearn.preprocessing import  MinMaxScaler
import pandas as pd
import os

#---------------functions
def add2results(addr, model_name, model_param,  precision, recall, f1,
                tn, fp, fn, tp, tpr, fpr, fdr, fnr, tnr, train_time, test_time, description):
    new_row = {'model_name': [model_name],
               'precision': [precision], 'recall': [recall], 'f1_score': [f1],
               'tn': [tn], 'fp': [fp], 'fn': [fn], 'tp': [tp], 'tpr': [tpr], 'fpr': [fpr],
               'fdr': [fdr], 'fnr': [fnr], 'tnr': [tnr],
               'train_time': [train_time],
               'test_time': [test_time], 'model_param': [model_param], 'description': [description]}
    if os.path.exists(addr):
        pd.DataFrame(new_row).to_csv(addr, mode='a', index=False, header=False)
    elif not os.path.exists(addr):
        pd.DataFrame(new_row).to_csv(addr, mode='a', index=False)

#----------prepration--------------
time_stamp = time.strftime("%d-%m-%Y-%H-%M-%S")
os.mkdir("/media/sf_GAN_output/red-teaming-10-Sep-2023/GAN- Red-team Project/results/rf_eval_" + time_stamp )
ds_addr_2 = "/media/sf_GAN_output/red-teaming-10-Sep-2023/GAN- Red-team Project/pcap/csv/"
ds_addr = "/media/sf_GAN_output/red-teaming-10-Sep-2023/4.comparison_with_CICIDSandUNSW/New DS/"
results_addr = "/media/sf_GAN_output/red-teaming-10-Sep-2023/GAN- Red-team Project/results/rf_eval_" + time_stamp + "/"
os.mkdir(results_addr + 'saved_model')
saved_model_addr = results_addr + 'saved_model/'
train_list = [
    ds_addr_2 + "labeled_sc-01-beacon_RCE_exfil_5mb.csv",

            #
            ds_addr + "labeled_sc-6_normal_tr.csv",

            #ds_addr +"labeled_sc-10_filtered_Normal_general-scans.csv",

            ds_addr +"labeled_sc-7_filtered_Normal_exploit-postgresQL-FTP-samba.csv",

            ds_addr + "labeled_attack_http_exfil_filtered.csv",
            ds_addr + "labeled_sc-8_filtered_Normal_exploit_win_ICMP-exfil_SMTP.csv",
            ds_addr + "labeled_sc-3_filtered_Normal_brute-ssh_exfil-SCP.csv",
            ds_addr + "labeled_sc-4_filtered_Normal_brute-telnet_exfil-FTP.csv",
            ds_addr + "labeled_attack_dns_exfil_filtered.csv",

    #ds_addr +"labeled_http_beaconig_9_bots_and_cmd.csv",
            #ds_addr +"labeled_http_beaconig_9_bots.csv" ,
            #ds_addr +"labeled_attack_ddos_filtered.csv" ,
            #

]
test_list = [
    ds_addr +"labeled_sc-5_filtered_Normal_exploit-http-smb-netbios-ssn.csv",
    ds_addr + "labeled_sc-12_filtered_gradual_CD_bruteforce-SSH_Exfil-C2.csv",
    ds_addr + "labeled_sc-15_filtered_gradual_CD_bruteforce-telnet_Exfil-FTP.csv",
    ds_addr +"labeled_sc-5_filtered_Normal_exploit-http-smb-netbios-ssn.csv",
    #ds_addr +"labeled_sc-9_filtered_normal_tr_lat_mov_DDoS.csv",
    ds_addr_2 +"labeled_sc_03_beacon4prob.csv",
    #ds_addr + "labeled_sc-16_filtered_periodic_CD_without_attack.csv",
    #ds_addr + "labeled_sc-11_filtered_Normal_web-browsing.csv",
    #ds_addr + "labeled_sc-16_filtered_periodic_CD_without_attack.csv",
       # ds_addr + "labeled_sc-2_normal_tr_dns-exfil.csv",
       # ds_addr +"labeled_sc-1_filtered_Normal_bruteforce_exfiltrate-C2-http.csv",
       # ds_addr +"labeled_http_beaconig_9_bots_and_cmd.csv",
       # ds_addr +"labeled_http_beaconig_9_bots.csv" ,
       # ds_addr_2 + "labeled_sc-02-randomize-beacon_RCE_exfil_5mb.csv"

    #ds_addr + "labeled_sc-14_filtered_web_traffic-wget-.csv"


]
#benign_only_list = ["labeled_sc-14_filtered_web_traffic-wget-.csv", "labeled_sc-11_filtered_Normal_web-browsing.csv"]
eval= Evaluation()
rf = rf_model()
utile = Utils()
ds= Dataset()
scale = MinMaxScaler() # we did not need pipeline for this

#-------printing
print('-----------Training Sets------------')
for item in train_list:
    df = pd.read_csv(item)
    print(item.removeprefix(ds_addr_2), utile.return_duplicates(df))
print('-----------testing Sets------------')
for item in test_list:
    df = pd.read_csv(item)
    print(item.removeprefix(ds_addr_2), utile.return_duplicates(df))

print('--------------Combining DS for Train/Test-------------')
train = pd.concat(ds.load_multiple_DS(train_list))#.sample(1000)
ds.print_info(train)
test_combined = pd.concat(ds.load_multiple_DS(test_list))#.sample(1000)
ds.print_info(test_combined)

#---scaling
train_scaled = scale.fit_transform(train.loc[:, train.columns != 'label'])
"""
#----tune
print('----------Tuning------------')
_param= rf.tune(train_x= train_scaled, train_y= train.loc[:, train.columns == 'label'].values.ravel(), cv=2)
rf.set_params(**_param)
#----train
print('----------Training------------')
train_start = time.time()
rf.fit(X=train_scaled,y=train.loc[:, train.columns == 'label'].values.ravel() )
train_stop = time.time()

#------save model
pickle.dump(rf,open(saved_model_addr+ '_model', 'wb'))
"""
with open("/media/sf_GAN_output/red-teaming-10-Sep-2023/GAN- Red-team Project/results/rf_eval_22-09-2023-17-10-16/saved_model/_model", 'rb') as f:
    rf = pickle.load(f)

#-------test
print('----------Testing------------')

for i, item in enumerate(test_list):

    print("Test scenario name: {}".format(item.removeprefix(ds_addr)))
    test = ds.load_test_ds(item)#.sample(frac=.3)
    test_scaled = scale.transform(test.loc[:, test.columns != 'label'])
    test_start = time.time()
    xyhat = rf.predict(test_scaled)
    test_stop = time.time()
    xyhat_prob = rf.predict_proba(test_scaled)
    if len (test['label'].value_counts()) >1 :
        precision,recall,f1,tn,fp,fn,tp,tpr,fpr,fdr,fnr,tnr=eval.precision_recall_f1_tn_fp_fn_tp_tpr_fpr_fdr_fnr_tnr(y_predict=xyhat, y_true=test.loc[:, test.columns == 'label'].values.ravel())
        print('precision {} recall {} f1 {} tn {} fp {} fn {} tp {} tpr {} fpr {} fdr {} fnr {} tnr {} \n'.format(precision,
                                                                                                           recall, f1,
                                                                                                           tn,
                                                                                                           fp, fn, tp,
                                                                                                           tpr, fpr,
                                                                                                           fdr,
                                                                                                           fnr, tnr))
        add2results(results_addr + 'detection_compare.csv','rf', '--', precision, recall, f1, tn, fp, fn, tp, tpr, fpr, fdr, fnr, tnr,
                    train_time='--', test_time=test_stop-test_start, description='Test DS is: ' + item.removeprefix(ds_addr_2))
    elif len (test['label'].value_counts()) == 1:
        tn, fp, fn, tp,fnr,tnr = eval.tn_fp_fn_tp_fnr_tnr(y_predict=xyhat, y_true=test.loc[:, test.columns == 'label'].values.ravel())
        add2results(results_addr + 'detection_compare.csv', 'rf', '--', '-', '--', '--', tn, fp, fn, tp,
                    '--', '--', '--', fnr, tnr,
                    train_time='--', test_time=test_stop - test_start,
                    description='Test DS is: ' + item.removeprefix(ds_addr_2))
    if  not (item == ds_addr_2 + "labeled_sc-02-randomize-beacon_RCE_exfil_5mb.csv" ):
        pd.DataFrame(xyhat).to_csv(results_addr + item.removeprefix(ds_addr)+'_xyhat.csv', index=False)
        pd.DataFrame(xyhat_prob).to_csv(results_addr + item.removeprefix(ds_addr)+'_xyhat_proba.csv', index=False)
    elif item == ds_addr_2 + "labeled_sc-02-randomize-beacon_RCE_exfil_5mb.csv":
        pd.DataFrame(xyhat).to_csv(results_addr+ item.removeprefix(ds_addr_2)+'_xyhat.csv', index=False)
        pd.DataFrame(xyhat_prob).to_csv(results_addr +item.removeprefix(ds_addr_2)+'_xyhat_proba.csv', index=False)





print("-------------------Start Testing Combined Scenarios----------------")
test_scaled = scale.transform(test_combined.loc[:, test_combined.columns != 'label'])
test_combined_start = time.time()
xyhat_combined = rf.predict(test_scaled)
test_combined_stop = time.time()

xyhat_combine_proba = rf.predict_proba(test_scaled)
precision,recall,f1,tn,fp,fn,tp,tpr,fpr,fdr,fnr,tnr= eval.precision_recall_f1_tn_fp_fn_tp_tpr_fpr_fdr_fnr_tnr(y_predict=xyhat_combined, y_true=test_combined.loc[:, test_combined.columns == 'label'].values.ravel())
print('precision {} recall {} f1 {} tn {} fp {} fn {} tp {} tpr {} fpr {} fdr {} fnr {} tnr {}'.format(precision, recall, f1, tn,
                                                                                              fp, fn, tp, tpr, fpr, fdr,
                                                                                              fnr, tnr))

add2results(results_addr + 'detection_compare.csv', 'rf', '--', precision, recall, f1, tn, fp, fn, tp, tpr, fpr,
            fdr, fnr, tnr,
            train_time='--', test_time=test_combined_stop - test_combined_start, description='Combined individual Test DS')
pd.DataFrame(xyhat_combined).to_csv(results_addr + 'combined_test_xyhat.csv', index=False)
pd.DataFrame(xyhat_combine_proba).to_csv(results_addr +  'combined_test_xyhat_proba.csv', index=False)
#"""