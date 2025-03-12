
from library.build_ids_script import build_ids
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import numpy as np
#------load config
config_addr = yaml.safe_load("/home/mehrdad/PycharmProjects/C2_communication/nids/ids_config.yaml")
try:
    with open(config_addr, 'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
     print(f"An error occurred: {e}")
#--------------report addr
report2save= "/home/mehrdad/PycharmProjects/C2_communication/nids/train-test/"

#-----load ds
df = pd.read_csv(config["redteam_ds"])#.sample(1000)
x = df.loc[:, config["features"]]
X_train, X_test, y_train, y_test = train_test_split(x.loc[:, x.columns != "label"], x.loc[:, x.columns == "label"].values.ravel(),stratify=x.loc[:, x.columns == "label"].values.ravel()
                                                    , test_size=0.2, random_state=42)
print(f" X_train : {X_train.shape} X_test: {X_test.shape} y_tarin: {y_train.shape} y_test {y_test.shape}")
print(f" y_train : Benign {np.count_nonzero(y_train == 0)} Malicous {np.count_nonzero(y_train == 1)} ///  y_test: Benign {np.count_nonzero(y_test == 0)} Malicous: {np.count_nonzero(y_test == 1)}")

#----------------train and test
ids_list = ["cnn", "rf", "knn"]
for ids_name in ids_list: 
    ids = build_ids()
    ids.ids_build_tune_train(f"{ids_name}", train_ds_name=f"deepred-auto-c2", train_x=X_train,
        y_train= y_train,
        addr2save=report2save ,
        model2save=True)
    ids.ids_predict(f"{ids_name}", test_x=X_test, test_y=y_test,
                    save_predict=True, eval2save=True, test_ds_name=f"deepred-auto-c2", save_model=False, train_x_name= "deepred-auto-c2")
    