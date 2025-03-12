from build_ids.build_ids_script import build_ids

model = build_ids(config_addr='/home/mehrdad/PycharmProjects/C2_communication/build_ids/config.yaml')
model.ids_train_evaluate_kfold('rf', eval2save=True)
