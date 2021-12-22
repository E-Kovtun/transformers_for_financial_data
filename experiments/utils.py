import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import re
import spacy

from copy import deepcopy

from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import tqdm
from sklearn.model_selection import GroupKFold

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorboard.backend.event_processing import event_accumulator

import pytorch_lightning as pl

import datetime

from torch.profiler import profile, record_function, ProfilerActivity

# --------------------------- Data reading and some preprocessing ---------------------------

def read_preprocessed_financial_data(data_folder, enc_cols=[]):
    transactions = pd.read_csv(data_folder+'transactions.csv', sep=',')
    tr_types = pd.read_csv(data_folder+'tr_types.csv', sep=';')
    tr_mcc_codes = pd.read_csv(data_folder+'tr_mcc_codes.csv', sep=';')
    gender_train = pd.read_csv(data_folder+'gender_train.csv', sep=',')
    
    df = pd.merge(transactions, gender_train, on='customer_id', how='outer')
    df = pd.merge(df, tr_mcc_codes, on='mcc_code', how='outer') 
    df = pd.merge(df, tr_types, on='tr_type', how='outer') 
    
    df = df[~np.isnan(df['gender'])]
    
    le = LabelEncoder()
    df['term_id'] = le.fit_transform(df['term_id'])
    
    df['week'] = df['tr_datetime'].str.split(' ').apply(lambda x: int(x[0]) // 7)
    times = df['tr_datetime'].apply(lambda x: x.split(' ')[1].split(':'))
    days = df['tr_datetime'].apply(lambda x: int(x.split(' ')[0]) * 24 * 60 * 60)
    to_seconds = lambda x: int(x[0]) * 3600 + int(x[1]) * 60 + int(x[2])
    df['tr_datetime'] = times.apply(to_seconds) * days    
    
    df['mcc_description'] = df['mcc_description'].fillna(df['mcc_description'].value_counts().index[0])
    df['tr_description'] = df['tr_description'].fillna(df['tr_description'].value_counts().index[0])
    
    df['mcc_description'] = df['mcc_description'].apply(lambda x: re.sub('[^а-яА-Я\s]', '', x.lower()))
    df['tr_description'] = df['tr_description'].apply(lambda x: re.sub('[^а-яА-Я\s]', '', x.lower()))
    
    if len(enc_cols) > 0:
        nlp = spacy.load('ru_core_news_lg')
        
        for col in enc_cols:
            col_dict = {x: nlp(x).vector for x in df[col].unique()}
            df[col] = df[col].map(col_dict)
    
    return df

# --------------------------- Data preparation for models' training ---------------------------

class DatasetSamples(Dataset):
    def __init__(self, data, targets):
        self.samples = data
        self.targets = targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        target = self.targets[idx]
        
        return sample, target

    
def create_dataset(df, features, batch_size, shuffle=True):
    data_samples = []
    targets = []
    desc_cols = ['mcc_description', 'tr_description']
    feat = list(set(features) - set(desc_cols))
    desc_cols_feat = list(set(features) & set(desc_cols))
    
    for client in df['customer_id'].unique():
        df1 = df[df['customer_id'] == client]
        for i in range(len(df1)):
            data_sample = torch.cat(
                [
                    torch.tensor([df1[col].iloc[i]]).T for col in feat
                ], dim=1
            )
            if len(desc_cols_feat) > 0:
                data_sample_desc = torch.cat(
                    [
                        torch.tensor(df1[col].iloc[i]) for col in desc_cols_feat
                    ], dim=1
                )
                data_sample = torch.cat(
                    [
                        data_sample, data_sample_desc
                    ], dim=1
                )
            data_samples.append(data_sample)
            targets.append(df1['gender'].iloc[i][0])
        
            
    data_samples = nn.utils.rnn.pad_sequence(tuple(data_samples))

    dataset = DatasetSamples(data_samples, targets) 
    return dataset

# --------------------------- Models ---------------------------
    
class RNN(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_rnn_size=16, base_type="LSTM"):
        super(RNN, self).__init__()

        self.hidden_rnn_size = hidden_rnn_size
        self.base_type = base_type

        if self.base_type == "LSTM":
            self.rnn = nn.LSTM(input_size, self.hidden_rnn_size, batch_first=True)
        elif self.base_type == "GRU":
            self.rnn = nn.GRU(input_size, self.hidden_rnn_size, batch_first=True)
        elif self.base_type == "RNN":
            self.rnn = nn.RNN(input_size, self.hidden_rnn_size, batch_first=True)
        else:
            raise ValueError("Choose LSTM, or GRU, or RNN type")

        self.linear = nn.Linear(self.hidden_rnn_size, output_size)

    def forward(self, input_samples):
        h_n = input_samples
        if self.base_type == "LSTM":
            output, (h_n, c_n) = self.rnn(input_samples)
        elif self.base_type == "GRU" or self.base_type == "RNN":
            output, h_n = self.rnn(input_samples)

        output = self.linear(torch.squeeze(h_n))
        output = torch.sigmoid(output)

        return output
    
    
class ClassificationModel(pl.LightningModule):
    def __init__(self, model, train_data, test_data, batch_size=64, learning_rate=1e-3):
        super(ClassificationModel, self).__init__()
        self.model = model

        self.batch_size = batch_size
        self.loss_function = nn.BCELoss()

        self.train_data = train_data
        self.val_data = test_data

        self.learning_rate = learning_rate

    def forward(self, inputs):
        return self.model(inputs)

    @staticmethod
    def calculate_metrics(target, y_pred):
        target = target.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        acc = accuracy_score(target, y_pred > 0.5)

        try:
            roc_auc = roc_auc_score(target, y_pred)
        except ValueError:
            roc_auc = acc
        pr_auc = average_precision_score(target, y_pred)

        return acc, roc_auc, pr_auc

    def training_step(self, batch, batch_idx):
        sample, target = batch
        pred = self.forward(sample.float())

        train_loss = self.loss_function(pred.squeeze(), target.float())
        train_accuracy = (target == (pred.squeeze() > 0.5)).float().mean()

        self.log("train_loss", train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc", train_accuracy, on_step=False, on_epoch=True, prog_bar=True
        )

        return train_loss

    def validation_step(self, batch, batch_idx):
        sample, target = batch
        pred = self.forward(sample.float())

        val_loss = self.loss_function(pred.squeeze(), target.float())
        val_accuracy = (target == (pred.squeeze() > 0.5)).float().mean()

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", val_accuracy, prog_bar=True)

        return {
            "val_loss": val_loss,
            "val_acc": val_accuracy,
            "val_target": target,
            "val_predictions": pred,
        }

    def validation_epoch_end(self, outputs):
        predictions = torch.cat([x["val_predictions"] for x in outputs])
        target = torch.cat([x["val_target"] for x in outputs])

        accuracy, roc_auc, pr_auc = self.calculate_metrics(
            target.squeeze(), predictions.squeeze()
        )

        log_dict = {
            "mean_accuracy": accuracy,
            "mean_roc_auc": roc_auc,
            "mean_pr_auc": pr_auc,
        }

        for k, v in log_dict.items():
            self.log(k, v, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return opt

    def train_dataloader(self):
        train_dataloader = torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=4
        )
        return val_dataloader

# --------------------------- Models' evaluation ---------------------------

def calculate_metrics(target, y_pred):
    acc = accuracy_score(target, y_pred > 0.5)

    try:
        roc_auc = roc_auc_score(target, y_pred)
    except ValueError:
        roc_auc = acc
    pr_auc = average_precision_score(target, y_pred)

    return acc, roc_auc, pr_auc


def test_model(model, dataset):
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    test_predictions = []
    test_targets = []
    # get predictions on each batch
    for batch, target in test_loader:
        pred = model(batch)
        test_predictions.extend(pred.flatten().detach().cpu().numpy())
        test_targets.extend(target.flatten().detach().cpu().numpy())

    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)

    # calculate metrics
    metrics = calculate_metrics(test_targets, test_predictions)

    return test_predictions, test_targets, metrics

# --------------------------- Plotting logs and results ---------------------------

def prepare_one_log(ea, tag):
    logs = ea.Scalars(tag)
    logs = [logs[x].value for x in range(len(logs))]
    return logs


def prepare_logs(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    logs_tags = ["train_loss", "val_loss", "train_acc", "val_acc"]

    dict_logs = {}

    for tag in logs_tags:
        dict_logs[tag] = prepare_one_log(ea, tag)
    return dict_logs


def plot_train_process(log_dir):
    plt.rc('font', size=22)          # controls default text sizes
    plt.rc('axes', titlesize=30)     # fontsize of the axes title
    plt.rc('axes', labelsize=30)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=30)    # fontsize of the tick labels
    plt.rc('legend', fontsize=30)    # legend fontsize
    plt.rc('figure', titlesize=22)   # fontsize of the figure title 

    dict_logs = prepare_logs(log_dir)
    
    # plot losses
    plt.figure(figsize=(24, 8))
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.plot(dict_logs["train_loss"], label="Train")
    plt.plot(dict_logs["val_loss"], label="Validation")
    plt.legend(loc="center right")
    
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plt.plot(dict_logs["train_acc"], label="Train")
    plt.plot(dict_logs["val_acc"], label="Validation")
    plt.legend(loc="center right")
    
    return dict_logs

# --------------------------- Cross-validation function ---------------------------

def time_memory_consumption(data, path):
    with open(path, 'a') as f:
        f.write(data)

def cross_validation(df_week, base_type, base_model, features, group_col='customer_id', n_splits=5, epochs=100):
    group_kfold = GroupKFold(n_splits=n_splits)
    metrics = {'Accuracy': [], 'ROC AUC': [], 'PR AUC': []}
    for train_index, test_index in group_kfold.split(X=df_week, groups=df_week[group_col]):
        train_data = df_week.iloc[train_index]
        test_data = df_week.iloc[test_index]

        random.seed(123)
        val_customer_id = random.sample(train_data['customer_id'].unique().tolist(), int(train_data['customer_id'].nunique() * 0.3))
        val_data = train_data[train_data['customer_id'].isin(val_customer_id)]
        train_data = train_data[~train_data['customer_id'].isin(val_customer_id)]

        train_dataset = create_dataset(train_data, features, batch_size=64, shuffle=True)
        val_dataset = create_dataset(val_data, features, batch_size=64, shuffle=False)
        test_dataset = create_dataset(test_data, features, batch_size=64, shuffle=False)    

        model = ClassificationModel(
            model=base_model,
            train_data=train_dataset,
            test_data=val_dataset
        )
        
        current_time = datetime.datetime.now().strftime("%m%d%Y_%H:%M:%S")
        experiment_name = base_type + "_" + str(epochs) + "_" + current_time

        logger = pl.loggers.TensorBoardLogger(save_dir='../logs/', name=experiment_name)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath=f'../logs/{experiment_name}',
            filename='{epoch:02d}-{val_loss:.3f}',
            mode='min')

        trainer = pl.Trainer(
            max_epochs=epochs, 
            gpus=[0],  
            benchmark=True, 
            check_val_every_n_epoch=1, 
            logger=logger,
            callbacks=[checkpoint_callback])

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
            with record_function("model_training"):
                trainer.fit(model)
        torch.save(model.model.state_dict(), '../models/{}.pth'.format(experiment_name))
        time_memory_consumption(prof.key_averages().table(), '../models/training_{}.txt'.format(experiment_name))
        print(prof.key_averages().table())
        
        dict_logs = plot_train_process(logger.log_dir)
        plt.show()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
            with record_function("model_testing"):
                test_predictions, test_targets, metric = test_model(model, test_dataset)
        time_memory_consumption(prof.key_averages().table(), '../models/testing_{}.txt'.format(experiment_name))
        print(prof.key_averages().table())
        metrics['Accuracy'].append(metric[0])
        metrics['ROC AUC'].append(metric[1])
        metrics['PR AUC'].append(metric[2])
        
    return metrics