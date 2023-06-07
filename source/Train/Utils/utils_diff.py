import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import scipy
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
import random
import os
import copy
import math
import gc
import h5py
from obspy.imaging.spectrogram import spectrogram


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

def seed_everything(seed):
    """It sets all the seeds for reproducibility.

    Args:
    ----------
    seed : int
        Seed for all the methods
    """
    print("Setting seeds")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def DataPreprocessing(path, dataToProcess, when, preprocessing=False):
    """It preprocesses data .hdf5 and metadata .csv, creating a single file that 
    can be used during ML procedure.

    Args:
    ----------
    path : String
        Path where to save the preprocessed data. Raw data should be in a folder
        in path called "raw_data"
    dataToProcess : String
        "mini" or "all" or single station (e.g. "T1213" or "NRCA"...)
    when : String
        "pre" if you want to process pre data (label= [1, 0]), "post" if you want post data (label= [0, 1]), 
        "visso" if you want data between Visso EQ (Mw 5.9, October 26th 2016) and Norcia EQ (Mw 6.5, October 30th 2016) (label= [0, 0])
    """
    print("Creating subdatasets...")
    file_name = path+"\\raw_data\\"+dataToProcess+"_catalog_"+when+".hdf5"
    csv_file = path+"\\raw_data\\"+dataToProcess+"_attributes_"+when+".csv"

    # reading the csv file into a dataframe:
    df = pd.read_csv(csv_file)
    print(f'total events in csv file: {len(df)}')
    ev_list = df['trace_name'].to_list()

    print("Creating subdatasets...")
    start=0 
    trace_df = pd.DataFrame(columns=['E_channel','N_channel','Z_channel','trace_name','label'])
    # retrieving selected waveforms from the hdf5 file: 
    dtfl = h5py.File(file_name, 'r')
    for c, evi in enumerate(ev_list[start:], start=start):
          dataset = dtfl.get(str(evi)) 
          data = np.array(dataset)#.transpose()
          if len(data)<3:
            print("row "+str(c)+"incomplete")
          else:
            trace_df.at[int(c), "E_channel"] = data[0]
            trace_df.at[int(c), "N_channel"] = data[1]
            trace_df.at[int(c), "Z_channel"] = data[2]
            trace_df.at[int(c), "trace_name"] = evi
            if when=="pre":
              trace_df.at[int(c), "label"] = [1, 0]
            elif when=="visso":
              trace_df.at[int(c), "label"] = [0, 0]
            elif when=="post":
              trace_df.at[int(c), "label"] = [0, 1]
            
          if c%1000==0:
            print(c)
          # if c==end:
          #   break
    whole_df = pd.merge(trace_df, df, on=['trace_name'])
    file_name_save=path+"\\dataframe\\dataframe_"+when+"_"+dataToProcess+".csv"
    whole_df.to_pickle(file_name_save)
    del whole_df
    gc.collect()


def normalize_df_column(df_column):
    """ This returns the normalized version of a df columns
    """
    dataset=pd.DataFrame(df_column.to_list())
    set = dataset.iloc[:, 0:dataset.shape[1]].values
    set = np.expand_dims(set, axis=2)
    src= torch.from_numpy(set)
    # it normalizes each event independently
    src_max = src.reshape(src.shape[0],-1).abs().max(dim=1)[0].reshape(-1,1)
    src_norm = src.reshape(src.shape[0],-1)/src_max
    src = src_norm.reshape(src.shape).numpy()
    
#     Then in the main you do:
#     norm_df = pd.DataFrame(columns=['E_channel_norm','N_channel_norm','Z_channel_norm','trace_name'])
#     norm_E=normalize_df_column(df["E_channel"])
#     norm_N=normalize_df_column(df["N_channel"])
#     norm_Z=normalize_df_column(df["Z_channel"])
#     for index, row in df.iterrows():
#       norm_df.at[index, "trace_name"] = row['trace_name']
#       norm_df.at[index, "E_channel_norm"] = norm_E[index].flatten()
#       norm_df.at[index, "N_channel_norm"] = norm_N[index].flatten()
#       norm_df.at[index, "Z_channel_norm"] = norm_Z[index].flatten()
#     df = pd.merge(norm_df, df, on=['trace_name'])

    return src

def pre_post_equal_length(df_pre, df_visso, df_post, force_in_test, num_classes):
    """ This function is to ensure to have a balanced data. We remove events 
    from the bigger dataset, in order to have the same number of events pre and post Norcia.
    To make sure to remove random events in time, it first shuffles the rows
    Args:
    ----------
    df_pre : DataFrame
        Input DataFrame pre from where to eventually remove to make the dataset balanced
    df_visso : DataFrame
        Input DataFrame visso from where to eventually remove to make the dataset balanced. It this is empty it won't count.
    df_post : DataFrame
        Input DataFrame post from where to eventually remove to make the dataset balanced
    force_in_test : list
        traces to be forced in test set
    num_classes : int
        number of total classes we eant to split the df (pre, post and eventually visso, if num_classes==9)

    Returns:
    ----------
    df_pre : resulting DataFrame that is the shuffled version of the original one, and have the same number of events as df_post
    df_post : resulting DataFrame that is the shuffled version of the original one, and have the same number of events as df_pre
    """
    df_pre=df_pre.sample(frac=1).reset_index(drop=True)
    df_visso=df_visso.sample(frac=1).reset_index(drop=True)
    df_post=df_post.sample(frac=1).reset_index(drop=True)

    # check if traces in force_in_test belong to current station
    for i in force_in_test:
        trace_station=i.split('.')[0]
        df_station=df_pre['trace_name'][0].split('.')[0]
        if trace_station==df_station: # trace i in force_in_test belong to current station
            if i in df_pre['trace_name'].values: # trace i is in pre
                #move row at the beginning of df, so that we are sure we won't cut it
                index_to_shift=df_pre.loc[df_pre['trace_name'] == i].index[0]
                idx = df_pre.index.tolist()
                idx.remove(index_to_shift)
                df_pre = df_pre.reindex([index_to_shift] + idx)
            elif i in df_visso['trace_name'].values: # trace i is in visso
                #move row at the beginning of df, so that we are sure we won't cut it
                index_to_shift=df_visso.loc[df_visso['trace_name'] == i].index[0]
                idx = df_visso.index.tolist()
                idx.remove(index_to_shift)
                df_visso = df_visso.reindex([index_to_shift] + idx)
            else: # trace i is in post       
                #move row at the beginning of df, so that we are sure we won't cut it
                index_to_shift=df_post.loc[df_post['trace_name'] == i].index[0]
                idx = df_post.index.tolist()
                idx.remove(index_to_shift)
                df_post = df_post.reindex([index_to_shift] + idx) 
        # else: trace i doesn't belong to current station, so cut df without caring

    # compute length 
    len_class=0
    if num_classes==9: # this includes visso
        if len(df_visso)*4>len(df_pre) or len(df_visso)*4>len(df_post):
            if len(df_pre)<len(df_post):
                len_class=int(len(df_pre)/4)*4
            else:
                len_class=int(len(df_post)/4)*4
        else:
            len_class=len(df_visso)*4             
        df_visso=df_visso[:int(len_class/4)] 
    else: # this doesn't include visso
        if len(df_pre)<len(df_post):
            len_class=len(df_pre)
        else:
            len_class=len(df_post)
    df_pre=df_pre[:len_class] 
    df_post=df_post[:len_class]

#     print("len_class",len_class)
#     print("len(df_pre)",len(df_pre))
#     print("len(df_visso)",len(df_visso))
#     print("len(df_post)",len(df_post))
    return df_pre, df_visso, df_post

def frames_N_classes(df,num_classes, pre_or_post):
    """It takes a row from a df and it computes the difference in seconds between the event in the input row and the main. 
    This is called Time To Failure (TTF)
    Args:
    ----------
    df : DataFrame
        Input DataFrame pre or post from where to recompute classes
    num_classes : int
        number of total classes we eant to split the df (pre, post and eventually visso, if num_classes==9)
    pre_or_post : String
        "pre" or "post" or "visso". It's used to properly assign the new label
    Returns:
    ----------
    frames : list of (int(num_classes/2)) sub-DataFrames from the original df
        
    """

    df = df.rename(columns={'label': 'label_2classes'})
    df.sort_values(by='trace_start_time', inplace=True)
    if pre_or_post=="visso":
        N = len(df)
        frames = [ df.iloc[i*N:(i+1)*N].copy() for i in range(1)]
        for f in range(0,len(frames)):
            frames[f] = frames[f].reset_index()
            label = pd.DataFrame(columns=['label'])
            for i in range(0,len(frames[f])):
                lab= [0] * num_classes # initialize label as a 0 array
                lab[int(num_classes/2)]=1 
                label.at[int(i), "label"] = lab
            frames[f] = frames[f].assign(label=label)
    elif pre_or_post=="pre" or pre_or_post=="post":
      N = int(len(df)/int(num_classes/2))
      frames = [ df.iloc[i*N:(i+1)*N].copy() for i in range(int(num_classes/2))]
      for f in range(0,len(frames)):
            frames[f] = frames[f].reset_index()
            label = pd.DataFrame(columns=['label'])
            for i in range(0,len(frames[f])):
                  lab= [0] * num_classes # initialize label as a 0 array
                  if pre_or_post=="pre": # assign 1 to the correct class
                        lab[f]=1 
                  elif pre_or_post=="post":
                        if num_classes==9:
                              lab[int(num_classes/2)+f+1]=1 # let's shift by 1 to leave the place for "visso" class
                        else:
                              lab[int(num_classes/2)+f]=1
                  else:
                        print("pre_or_post must be 'pre' or 'visso' or 'post'")
                  label.at[int(i), "label"] = lab
            frames[f] = frames[f].assign(label=label)
    else:
        print("pre_or_post must be 'pre' or 'visso' or 'post'")
    return frames

    

def add_TTF_in_sec(row):
  """It takes a row from a df and it computes the difference in seconds between the event in the input row and the main. 
  This is called Time To Failure (TTF)
  Args:
  ----------
  row : pandas.core.series.Series
        row from Input DataFrame where to add column TTF
  Returns:
  ----------
  difference : float of the amount of time in seconds between the event in the input row and the main one
        
  """
  time=row['source_origin_time']
  norcia_datetime= datetime.strptime('2016-10-30T07:40:17.000000Z', '%Y-%m-%dT%H:%M:%S.%fZ')
  difference = (time-norcia_datetime).total_seconds()
  return difference
def tapering(training_set, taper=True):
      if taper:
            for i in range(len(training_set)):
                  window = scipy.signal.tukey(len(training_set[i]), alpha=0.03, sym=True) 
                  wZT=window*training_set[i]
                  training_set[i]=wZT
      return training_set


def train_val_test_split(df, signal_start, signal_end, train_percentage=0.70, val_percentage=0.10, test_percentage=0.20, force_in_test=[]):
  """It takes the given df and it splits it in train,val,test and each one is further splitted in X (features: channels), y (label: [1,0]=pre or [0,1]=post), index to
  easily find the sample in the given df).

  Args:
  ----------
  df : pandas.core.frame.DataFrame
        Input DataFrame to be splitted
  train_percentage : float 0<=train_percentage<=1 (Default=0.7)
        Percentage of df length to use as training data.
  val_percentage : float 0<=val_percentage<=1 (Default=0.1)
        Percentage of df length to use as validation data.
  test_percentage : float 0<=test_percentage<=1 (Default=0.2)
        Percentage of df length to use as testing data.
        Note: train_percentage+val_percentage+test_percentage must be <1 and should be =1.
  force_in_test : list
        traces to be forced in test set

  Returns:
  ----------
  numpy.ndarray
        split in train,val,test and each one is further splitted in X (features: channels), y (label: [1,0]=pre or [0,1]=post), index to
        easily find the sample in the given df). 
        
  """

  if (train_percentage+val_percentage+test_percentage)>1:
      print("WARNING: train_percentage+val_percentage+test_percentage cannot be grater than 1")
  # this is to avoid having the same event in train and val/test.
  # If the dataset comes from a single station, this does nothing but shuffling data.
  if df["source_id"][0]!='':
      source_id_array=df.groupby(['source_id']).sum()
      source_id_array=source_id_array.index.to_numpy() 
      #np.random.seed(123) 
      source_id_test=[]
      for i in force_in_test:
            if i not in df['trace_name'].values:
                  print("WARNING: ", i," not in df. This will cause an error.")
            source_id_test.append(df.loc[df['trace_name'] == i]['source_id'].values[0])
      source_id_test=np.array(source_id_test)
      source_id_array = np.setdiff1d(source_id_array,source_id_test) # remove indexes that contains traces for testing
      np.random.shuffle(source_id_array)
      source_id_array = np.append(source_id_array, source_id_test, axis=0) # add indexes that contains traces for testing, so that we ensure they ends up in test set
      
      source_id_train=source_id_array[:int(len(source_id_array)*train_percentage)]
      source_id_val=source_id_array[int(len(source_id_array)*train_percentage):int(len(source_id_array)*(train_percentage+val_percentage))]
      source_id_test=source_id_array[int(len(source_id_array)*(train_percentage+val_percentage)):]

      train_df=df.loc[df['source_id'].isin(source_id_train)]
      train_df = train_df.sample(frac=1).reset_index(drop=True)
      val_df=df.loc[df['source_id'].isin(source_id_val)]
      val_df = val_df.sample(frac=1).reset_index(drop=True)
      test_df=df.loc[df['source_id'].isin(source_id_test)]
      test_df = test_df.sample(frac=1).reset_index()
  else:
      train_df=df[:int(len(df)*train_percentage)]
      train_df = train_df.sample(frac=1).reset_index(drop=True)
      val_df=df[int(len(df)*train_percentage):int(len(df)*(train_percentage+val_percentage))]
      val_df = val_df.sample(frac=1).reset_index(drop=True)
      test_df=df[int(len(df)*(train_percentage+val_percentage)):]
      test_df = test_df.sample(frac=1).reset_index()
      
  print("Events in train dataset: ",len(train_df))
  print("Events in validation dataset: ",len(val_df))
  print("Events in test dataset: ",len(test_df))
  df=pd.concat([train_df, val_df])
  df=pd.concat([df, test_df])
  df=df.reset_index(drop=True)

  train_size = len(train_df) #int(len(df) * train_percentage)
  val_size = len(val_df) #int(len(df) * val_percentage)
  test_size = len(test_df) #int(len(df) * test_percentage)
  
  print("Computing E channel")
  df_E_channel_norm=pd.DataFrame(df["E_channel"].to_list())
  dataset_trainE = df_E_channel_norm[0:train_size]
  training_setE = dataset_trainE.iloc[:, signal_start:signal_end].values
  training_setE=tapering(training_setE)
  training_setE = np.expand_dims(training_setE, axis=2)
  dataset_valE = df_E_channel_norm[train_size:train_size+val_size]
  val_setE = dataset_valE.iloc[:, signal_start:signal_end].values
  val_setE=tapering(val_setE)
  val_setE = np.expand_dims(val_setE, axis=2)
  dataset_testE = df_E_channel_norm[train_size+val_size:train_size+val_size+test_size]
  test_setE = dataset_testE.iloc[:, signal_start:signal_end].values
  test_setE=tapering(test_setE)
  test_setE = np.expand_dims(test_setE, axis=2)

  print("Computing N channel")
  df_N_channel_norm=pd.DataFrame(df["N_channel"].to_list())
  dataset_trainN = df_N_channel_norm[0:train_size]
  training_setN = dataset_trainN.iloc[:, signal_start:signal_end].values
  training_setN=tapering(training_setN)
  training_setN = np.expand_dims(training_setN, axis=2)
  dataset_valN = df_N_channel_norm[train_size:train_size+val_size]
  val_setN = dataset_valN.iloc[:, signal_start:signal_end].values
  val_setN=tapering(val_setN)
  val_setN = np.expand_dims(val_setN, axis=2)
  dataset_testN = df_N_channel_norm[train_size+val_size:train_size+val_size+test_size]
  test_setN = dataset_testN.iloc[:, signal_start:signal_end].values
  test_setN=tapering(test_setN)
  test_setN = np.expand_dims(test_setN, axis=2)

  print("Computing Z channel")
  df_Z_channel_norm=pd.DataFrame(df["Z_channel"].to_list())
  dataset_trainZ = df_Z_channel_norm[0:train_size]
  training_setZ = dataset_trainZ.iloc[:, signal_start:signal_end].values
  training_setZ=tapering(training_setZ)
  training_setZ = np.expand_dims(training_setZ, axis=2)
  dataset_valZ = df_Z_channel_norm[train_size:train_size+val_size]
  val_setZ = dataset_valZ.iloc[:, signal_start:signal_end].values
  val_setZ=tapering(val_setZ)
  val_setZ = np.expand_dims(val_setZ, axis=2)
  dataset_testZ = df_Z_channel_norm[train_size+val_size:train_size+val_size+test_size]
  test_setZ = dataset_testZ.iloc[:, signal_start:signal_end].values
  test_setZ=tapering(test_setZ)
  test_setZ = np.expand_dims(test_setZ, axis=2)

#   df_label=pd.DataFrame(df["label"].to_list())
#   dataset_trainlabel = df_label[0:train_size]
#   y_train = dataset_trainlabel.iloc[:, 0:dataset_trainlabel.shape[1]].values
#   dataset_vallabel = df_label[train_size:train_size+val_size]
#   y_val = dataset_vallabel.iloc[:, 0:dataset_vallabel.shape[1]].values
#   dataset_testlabel = df_label[train_size+val_size:train_size+val_size+test_size]
#   y_test = dataset_testlabel.iloc[:, 0:dataset_testlabel.shape[1]].values

  print("Computing index")
  df_index=pd.DataFrame(df.index.to_list())
  dataset_trainindex = df_index[0:train_size]
  print("dataset_trainindex.shape[1]",dataset_trainindex.shape[1])
  index_train = dataset_trainindex.iloc[:, 0:dataset_trainindex.shape[1]].values
  dataset_valindex = df_index[train_size:train_size+val_size]
  index_val = dataset_valindex.iloc[:, 0:dataset_valindex.shape[1]].values
  dataset_testindex = df_index[train_size+val_size:train_size+val_size+test_size]
  index_test = dataset_testindex.iloc[:, 0:dataset_testindex.shape[1]].values

  X_train=np.append(training_setE,training_setN, axis=2)
  X_train=np.append(X_train, training_setZ, axis=2)
  print("X_train.shape: ",X_train.shape)
  #print("y_train.shape: ",y_train.shape)
  print("index_train.shape: ",index_train.shape)
  X_val=np.append(val_setE,val_setN, axis=2)
  X_val=np.append(X_val, val_setZ, axis=2)
  print("X_val.shape: ",X_val.shape)
  #print("y_val.shape: ",y_val.shape)
  print("index_val.shape: ",index_val.shape)
  X_test=np.append(test_setE,test_setN, axis=2)
  X_test=np.append(X_test, test_setZ, axis=2)
  print("X_test.shape: ",X_test.shape)
  #print("y_test.shape: ",y_test.shape)
  print("index_test.shape: ",index_test.shape)
  
  return df, X_train, index_train, X_val, index_val, X_test, index_test
  #return df, X_train, y_train, index_train, X_val, y_val, index_val, X_test, y_test, index_test

# def create_dataloader(X, y, index, target_dataset, batch_size = 32):
#   """It takes the given numpy.ndarrays and it changes torch.utils.data.DataLoader, making data suitable for the model training.

#   Args:
#   ----------
#   X : numpy.ndarray
#         Model features: channels
#   y : numpy.ndarray
#         Model label: [post,pre]
#   index : numpy.ndarray
#         indexes: to easily find the sample in the given df
#   batch_size : int (Default=32)
#         Size of the batch used during the training of the model.
#   target_dataset: string
#         "train_dataset" or "val_dataset" or "test_dataset". This choice is to correctly select "shuffle"
#         and "drop_last" parameters in torch.utils.data.DataLoader function

#   Returns:
#   ----------
#   dl : torch.utils.data.DataLoader made of X,y,index

#   """
#   src, lab, idx = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32)), torch.from_numpy(index)
#   src=torch.nn.functional.normalize(src)
#   dataset = TensorDataset(src, lab, idx)
#   if target_dataset=="train_dataset":
#     dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
#   elif target_dataset=="val_dataset":
#     dl= torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
#   elif target_dataset=="test_dataset":
#     dl= torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
#   else:
#     print("target_dataset not valid.")
#   return dl

def create_dataloader(X, y, index, target_dataset, batch_size = 32, normalize_data=True):
  """It takes the given numpy.ndarrays and it changes torch.utils.data.DataLoader, making data suitable for the model training.

  Args:
  ----------
  X : numpy.ndarray
        Model features: channels
  y : numpy.ndarray
        Model label: [post,pre]
  index : numpy.ndarray
        indexes: to easily find the sample in the given df
  batch_size : int (Default=32)
        Size of the batch used during the training of the model.
  target_dataset: string
        "train_dataset" or "val_dataset" or "test_dataset". This choice is to correctly select "shuffle"
        and "drop_last" parameters in torch.utils.data.DataLoader function
  normalize_data: Boolean
        normalize or not data

  Returns:
  ----------
  dl : torch.utils.data.DataLoader made of X,y,index

  """
  src, lab, idx = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32)), torch.from_numpy(index)
  #src = src/src.reshape(-1,3).max(0, keepdim=True)[0] # <- it normalizes each channel independently
  #src = src/src.abs().max() # <- it normalizes on everything
  #src=torch.nn.functional.normalize(src) # <- it normalizes using torch function
  #bottom: it normalizes each event independently
  if normalize_data:
      src_max = src.reshape(src.shape[0],-1).abs().max(dim=1)[0].reshape(-1,1)
      src_norm = src.reshape(src.shape[0],-1)/src_max
      src = src_norm.reshape(src.shape)  

  dataset = TensorDataset(src, lab, idx)
  if target_dataset=="train_dataset":
    dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
  elif target_dataset=="val_dataset":
    dl= torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
  elif target_dataset=="test_dataset":
    dl= torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
  else:
    print("target_dataset not valid.")
  return dl


######## MODELS ###########

class FCN(nn.Module):
    def __init__(self, num_feature, num_class):
        super(FCN, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 32)
        self.layer_out = nn.Linear(32, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, batch_init, batch_size, steps_in):
        x = self.layer_1(batch_init.permute(0,2,1))#batch_init.view(batch_size,3,steps_in), forse era giusto view in questo caso
        x = self.relu(x)
        x = self.dropout(x)
        x_1=x.clone()

        x = self.layer_2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x_2=x.clone()

        x = self.layer_3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x_3=x.clone()

        x = self.layer_out(x)

        return x,x_1,x_2,x_3

class FCN_for_saliency(nn.Module):
    def __init__(self, num_feature, num_class):
        super(FCN_for_saliency, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 32)
        self.layer_out = nn.Linear(32, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, batch_init, batch_size, steps_in):
        x = self.layer_1(batch_init.permute(0,2,1))#batch_init.view(batch_size,3,steps_in)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)
        x=x.sum(dim=1)
        return x


class CNN(nn.Module): #This model is inspired by Temporal earthquake forecasting by Ong et al.
    def __init__(self, num_feature, num_class):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(3, 32, 7)
        self.BatchNorm1 =nn.BatchNorm1d(32) 
        self.conv2 = nn.Conv1d(32, 64, 7,stride=2)
        self.BatchNorm2 =nn.BatchNorm1d(64) 
        self.conv3 = nn.Conv1d(64, 64, 5)
        self.BatchNorm3 =nn.BatchNorm1d(64) 
        self.conv4 = nn.Conv1d(64, 64, 5)
        self.BatchNorm4 =nn.BatchNorm1d(64) 
        self.conv5 = nn.Conv1d(64, 128, 5)
        self.BatchNorm5 =nn.BatchNorm1d(128) 
        self.conv6 = nn.Conv1d(128, 256, 3)
        self.BatchNorm6 =nn.BatchNorm1d(256) 
        self.conv7 = nn.Conv1d(256, 256, 3)
        self.BatchNorm7 =nn.BatchNorm1d(256) 
        self.maxpool = nn.MaxPool1d(3,1) #kernel=3, stride=1
        self.relu = nn.ReLU() #activation relu modul
        self.dropout = nn.Dropout(0.3)
        self.flatten=nn.Flatten()
        self.AvgPool=nn.AvgPool1d(1215,1) 
        self.layer_out = nn.Linear(256, num_class) 
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, batch_init, batch_size, steps_in, softmax=False): 
        x = self.relu(self.maxpool(self.BatchNorm1(self.conv1(batch_init.permute(0,2,1))))) #batch_init.view(batch_size,3,steps_in)
        x_1=x.clone()
        x = self.relu(self.maxpool(self.BatchNorm2(self.conv2(x)))) 
        x_2=x.clone()
        x = self.relu(self.maxpool(self.BatchNorm3(self.conv3(x)))) 
        x_3=x.clone()
        x = self.relu(self.maxpool(self.BatchNorm4(self.conv4(x)))) 
        x_4=x.clone()
        x = self.relu(self.maxpool(self.BatchNorm5(self.conv5(x))))
        x_5=x.clone() 
        x = self.relu(self.maxpool(self.BatchNorm6(self.conv6(x))))
        x_6=x.clone() 
        x = self.relu(self.maxpool(self.BatchNorm7(self.conv7(x))))
        x_7=x.clone() 
        x = self.dropout(self.flatten(self.AvgPool(x)))
        x = self.layer_out(x)
        if softmax:
            x = self.softmax(x)
        return x,x_1, x_2,x_3,x_4,x_5,x_6,x_7


# ######## EVALUATION ###########


def to_one_hot(probVec):
  """It creates a one hot vector, representing the corresponding class,
  starting from the probabilty Vector predicted by the model.

  Args:
  ----------
  probVec : torch.Tensor
        Probabilty Vector predicted by the model
  Returns:
  ----------
  one_hot : one hot torch.Tensor array. This is 1 in the position of predicted class, 0 otherwise

  """
  idx = np.argmax(probVec, axis=-1)
  one_hot = torch.zeros(len(probVec))
  one_hot[idx] = 1
  return one_hot

def plot_input_allTrace_label(df,inputte,outputte,labeltte,indexte,inp_size,btch=20,start_wndw=1,end_wndw=20):
  """For each channel (E, N, Z), it plots the sequence used in input for the model, the whole trace from where
  the input sequence comes from and it prints the label together with the output of the model, specifying if the prediction is wrong.

  Args:
  ----------
  df : pandas.core.frame.DataFrame
        whole DataFrame used for this experiment. It contains train, validation and test data
  inputte : list of torch.Tensor
        list of all the torch.Tensors used as input for the model
  outputte : list of torch.Tensor
        list of all the torch.Tensors we have in output from the model
  labeltte : list of torch.Tensor
        list of all the torch.Tensors that are the label for the model
  indexte : list of torch.Tensor
        list of all the torch.Tensors that we save to match the same window in df and the DataLoader, by the use of the index
  inp_size : int
        length of each channel in one input sequence
  btch : int (Default=20)
        batch where to select windows that the user want to visualize
  start_wndw : int (Default=1)
        starting window in the selected batch from where the visualization begins
  end_wndw : int (Default=20)
        ending window in the selected batch until where the visualization stops

  """

  for wndw in range(start_wndw,end_wndw):
    if not torch.all(to_one_hot(outputte[btch][wndw].cpu()).eq(labeltte[btch][wndw].cpu())):
      print("wrong")
    plt.plot([r*0.01 for r in range(0,inp_size)], inputte[btch][wndw][:,0].cpu(), label=['E'])
    plt.xlabel('time (s)')
    plt.ylabel('waveform')
    plt.legend(loc='lower left')
    plt.show()

    plt.plot([r*0.01 for r in range(0,inp_size)], inputte[btch][wndw][:,1].cpu(), label=['N'])
    plt.xlabel('time (s)')
    plt.ylabel('waveform')
    plt.legend(loc='lower left')
    plt.show()

    plt.plot([r*0.01 for r in range(0,inp_size)], inputte[btch][wndw][:,2].cpu(), label=['Z'])
    plt.xlabel('time (s)')
    plt.ylabel('waveform')
    plt.legend(loc='lower left')
    plt.show()
    print('Output',outputte[btch][wndw].cpu())
    print('Output one_hot',to_one_hot(outputte[btch][wndw].cpu()))
    print('Label',labeltte[btch][wndw].cpu())


def plot_hidden(df,inputte,outputte,outputte_after1layer,outputte_after2layer,outputte_after3layer,outputte_lastlayer,
                labeltte,indexte,inp_size,btch=0,start_wndw=1,end_wndw=20,plot_hid=True):
  """Combining the three channels (E, N, Z), it plots the sequence used in input for the model, the intermediate output after
  each of the layer of the model, and the whole trace from where the input sequence comes from.
  It prints the label together with the output of the model, specifying if the prediction is wrong.
  Note: this funcion doesn't work for TF and CNN models. For these models this visualization is not implemented

  Args:
  ----------
  df : pandas.core.frame.DataFrame
        whole DataFrame used for this experiment. It contains train, validation and test data
  inputte : list of torch.Tensor
        list of all the torch.Tensors used as input for the model
  outputte : list of torch.Tensor
        list of all the torch.Tensors we have in output from the model
  outputte_after1layer : list of torch.Tensor
        list of all the torch.Tensors intermediate step we save within the model, after it first layer.
  outputte_after2layer : list of torch.Tensor
        list of all the torch.Tensors intermediate step we save within the model, after it second layer.
  outputte_after3layer : list of torch.Tensor
        list of all the torch.Tensors intermediate step we save within the model, after it third layer.
  outputte_lastlayer : list of torch.Tensor
        list of all the torch.Tensors intermediate step we save within the model, in the layer, before combining the classification per channel.
  labeltte : list of torch.Tensor
        list of all the torch.Tensors that are the label for the model
  indexte : list of torch.Tensor
        list of all the torch.Tensors that we save to match the same window in df and the DataLoader, by the use of the index
  inp_size : int
        length of each channel in one input sequence
  btch : int (Default=20)
        batch where to select windows that the user want to visualize
  start_wndw : int (Default=1)
        starting window in the selected batch from where the visualization begins
  end_wndw : int (Default=20)
        ending window in the selected batch until where the visualization stops
  plot_hid : Bool (Default=True)
        If true it plots the intermediate layers, otherwhise it just plot the input and the
        whole window from where the input comes from

  """
  c=0
  tot=0
  for wndw in range(start_wndw,end_wndw):
    tot=tot+1
    if not torch.all(to_one_hot(outputte[btch][wndw].cpu()).eq(labeltte[btch][wndw].cpu())):
      print("wrong")
      c=c+1
    plt.plot([r*0.01 for r in range(0,inp_size)], inputte[btch][wndw][:,0].cpu(), alpha=1.0, label=['E'])
    plt.plot([r*0.01 for r in range(0,inp_size)], inputte[btch][wndw][:,1].cpu(), alpha=0.8, label=['N'])
    plt.plot([r*0.01 for r in range(0,inp_size)], inputte[btch][wndw][:,2].cpu(), alpha=0.6, label=['Z'])
    plt.xlabel('time (s)')
    plt.ylabel('waveform')
    plt.legend(loc='lower left')
    plt.show()
    if plot_hid:
      plt.plot(outputte_after1layer[btch][wndw][0].cpu(), label=['E after1layer'], alpha=1.0)
      plt.plot(outputte_after1layer[btch][wndw][1].cpu(), label=['N after1layer'], alpha=0.8)
      plt.plot(outputte_after1layer[btch][wndw][2].cpu(), label=['Z after1layer'], alpha=0.6)
      plt.legend(loc='lower left')
      plt.show()
      plt.plot(outputte_after2layer[btch][wndw][0].cpu(), label=['E after2layer'], alpha=1.0)
      plt.plot(outputte_after2layer[btch][wndw][1].cpu(), label=['N after2layer'], alpha=0.8)
      plt.plot(outputte_after2layer[btch][wndw][2].cpu(), label=['Z after2layer'], alpha=0.6)
      plt.legend(loc='lower left')
      plt.show()
      plt.plot(outputte_after3layer[btch][wndw][0].cpu(), label=['E after3layer'], alpha=1.0)
      plt.plot(outputte_after3layer[btch][wndw][1].cpu(), label=['N after3layer'], alpha=0.8)
      plt.plot(outputte_after3layer[btch][wndw][2].cpu(), label=['Z after3layer'], alpha=0.6)
      plt.legend(loc='lower left')
      plt.show()
      plt.plot(outputte_lastlayer[btch][wndw][0].cpu(), label=['E last layer'], alpha=1.0)
      plt.plot(outputte_lastlayer[btch][wndw][1].cpu(), label=['N last layer'], alpha=0.8)
      plt.plot(outputte_lastlayer[btch][wndw][2].cpu(), label=['Z last layer'], alpha=0.6)
      plt.legend(loc='lower left')
      plt.show()

#     plt.plot([r*0.01 for r in range(0,(inp_size*10))],df['E_channel'].iloc[indexte[btch][wndw].item()], alpha=1.0, label=['all E'])
#     plt.plot([r*0.01 for r in range(0,(inp_size*10))],df['N_channel'].iloc[indexte[btch][wndw].item()], alpha=0.8, label=['all N'])
#     plt.plot([r*0.01 for r in range(0,(inp_size*10))],df['Z_channel'].iloc[indexte[btch][wndw].item()], alpha=0.6, label=['all Z'])
#     plt.xlabel('time (s)')
#     plt.ylabel('all waveform')
#     plt.legend(loc='lower left')
#     plt.show()

    print('Output',outputte[btch][wndw].cpu())
    print('Output one_hot',to_one_hot(outputte[btch][wndw].cpu()))
    print('Label',labeltte[btch][wndw].cpu())



def performances(df,outputte,labeltte,indexte,batch_size=32):
  """It computes model performances and save some values of the test set in lists and dictionaries for later statistics.

  Args:
  ----------
  df : pandas.core.frame.DataFrame
        whole DataFrame used for this experiment. It contains train, validation and test data
  outputte : list of torch.Tensor
        list of all the torch.Tensors we have in output from the model
  labeltte : list of torch.Tensor
        list of all the torch.Tensors that are the label for the model
  indexte : list of torch.Tensor
        list of all the torch.Tensors that we save to match the same window in df and the DataLoader, by the use of the index
  batch_size int (Default=32)
        Size of the batch used during the training of the model.

  Returns:
  ----------
  tot : int
        Amount of total traces in the test set
  wrong : int
        Amount of traces in the test set such that the correspinding class is predicted wrongly by the model
  tot_pre : int
        Amount of total traces in the test set, where the label is pre
  wrong_pre : int
        Amount of traces in the test set, where the label is pre, such that the model predict class="post"
  tot_post : int
        Amount of total traces in the test set, where the label is post
  wrong_post : int
        Amount of traces in the test set, where the label is post, such that the model predict class="pre"
#   magnitude_of_all : list
#         list of all magnitudes for the traces in the test set
#   magnitude_of_wrong_pre : list
#         list of magnitudes for the traces in the test set where the label is pre, but the model predict class="post"
#   magnitude_of_wrong_post : list
#         list of magnitudes for the traces in the test set where the label is post, but the model predict class="pre"
#   latitude_of_all : list
#         list of all latitudes for the traces in the test set
#   longitude_of_all : list
#         list of all longitudes for the traces in the test set
#   latitude_of_wrong_pre : list
#         list of latitudes for the traces in the test set where the label is pre, but the model predict class="post"
#   longitude_of_wrong_pre : list
#         list of longitudes for the traces in the test set where the label is pre, but the model predict class="post"
#   latitude_of_wrong_post : list
#         list of latitudes for the traces in the test set where the label is post, but the model predict class="pre"
#   longitude_of_wrong_post : list
#         list of longitudes for the traces in the test set where the label is post, but the model predict class="pre"
#   trace_start_time_of_all : list
#         list of all traces' start time for the traces in the test set
#   trace_start_time_of_wrong_pre : list
#         list of traces' start time for the traces in the test set where the label is pre, but the model predict class="post"
#   trace_start_time_of_wrong_post : list
#         list of traces' start time for the traces in the test set where the label is post, but the model predict class="pre"
  stationsDictTot : dict
        dictionary of number of all the events per station in test set
  stationsDictWrong : dict
        dictionary of number of all the events wrongly classified per station in test set
  """
  tot=0
  wrong=0
  tot_pre=0
  wrong_pre=0
  tot_post=0
  wrong_post=0
#   magnitude_all=[]
#   magnitude_of_wrong_pre=[]
#   magnitude_of_wrong_post=[]
#   distance_all=[]
#   distance_of_wrong_pre=[]
#   distance_of_wrong_post=[]
#   latitude_of_all=[]
#   longitude_of_all=[]
#   latitude_of_wrong_pre=[]
#   longitude_of_wrong_pre=[]
#   latitude_of_wrong_post=[]
#   longitude_of_wrong_post=[]
#   trace_start_time_of_all=[]
#   trace_start_time_of_wrong_pre=[]
#   trace_start_time_of_wrong_post=[]
#   snr_of_all=[]
#   snr_of_wrong_pre=[]
#   snr_of_wrong_post=[]
  stationsList=df[["receiver_name","E_channel"]].groupby(['receiver_name']).count().index.to_numpy() 
  stationsDictTot = dict.fromkeys(stationsList,0)
  stationsDictWrong = dict.fromkeys(stationsList,0)
  for w in range(len(outputte)):
    for b in range(0,batch_size):
      tot=tot+1
      # magnitude_all.append(df['source_magnitude'].iloc[indexte[w][b].item()])
      # distance_all.append(df['source_distance_km'].iloc[indexte[w][b].item()])
      # latitude_of_all.append(df['source_latitude'].iloc[indexte[w][b].item()])
      # longitude_of_all.append(df['source_longitude'].iloc[indexte[w][b].item()])
      # trace_start_time_of_all.append(df['trace_start_time'].iloc[indexte[w][b].item()])
      # snr_of_all.append(df['snr_db'].iloc[indexte[w][b].item()])
      key=df['receiver_name'].iloc[indexte[w][b].item()]
      stationsDictTot[key] = stationsDictTot[key] + 1 #if key in stationsDictTot else 1
      if not torch.all(to_one_hot(outputte[w][b].cpu()).eq(labeltte[w][b].cpu())):
        wrong=wrong+1
        stationsDictWrong[key] = stationsDictWrong[key] + 1 
      if labeltte[w][b][0]==0: #post
        tot_post=tot_post+1
        if not torch.all(to_one_hot(outputte[w][b].cpu()).eq(labeltte[w][b].cpu())):
          wrong_post=wrong_post+1
      #     magnitude_of_wrong_post.append(df['source_magnitude'].iloc[indexte[w][b].item()])
      #     distance_of_wrong_post.append(df['source_distance_km'].iloc[indexte[w][b].item()])
      #     latitude_of_wrong_post.append(df['source_latitude'].iloc[indexte[w][b].item()])
      #     longitude_of_wrong_post.append(df['source_longitude'].iloc[indexte[w][b].item()])
      #     trace_start_time_of_wrong_post.append(df['trace_start_time'].iloc[indexte[w][b].item()])
      #     snr_of_wrong_post.append(df['snr_db'].iloc[indexte[w][b].item()])

      else: #pre
        tot_pre=tot_pre+1
        
        if not torch.all(to_one_hot(outputte[w][b].cpu()).eq(labeltte[w][b].cpu())):
          wrong_pre=wrong_pre+1
      #     magnitude_of_wrong_pre.append(df['source_magnitude'].iloc[indexte[w][b].item()])
      #     distance_of_wrong_pre.append(df['source_distance_km'].iloc[indexte[w][b].item()])
      #     latitude_of_wrong_pre.append(df['source_latitude'].iloc[indexte[w][b].item()])
      #     longitude_of_wrong_pre.append(df['source_longitude'].iloc[indexte[w][b].item()])
      #     trace_start_time_of_wrong_pre.append(df['trace_start_time'].iloc[indexte[w][b].item()])
      #     snr_of_wrong_pre.append(df['snr_db'].iloc[indexte[w][b].item()])

  #return tot,wrong,tot_pre,wrong_pre,tot_post,wrong_post, magnitude_all, magnitude_of_wrong_pre,magnitude_of_wrong_post, latitude_of_all, longitude_of_all, latitude_of_wrong_pre, longitude_of_wrong_pre, latitude_of_wrong_post, longitude_of_wrong_post,trace_start_time_of_all, trace_start_time_of_wrong_pre, trace_start_time_of_wrong_post, stationsDictTot, stationsDictWrong#, distance_all, distance_of_wrong_pre, snr_of_all, snr_of_wrong_pre
  return tot,wrong,tot_pre,wrong_pre,tot_post,wrong_post, stationsDictTot, stationsDictWrong#, distance_all, distance_of_wrong_pre, snr_of_all, snr_of_wrong_pre


def feature_tot_wrong(feature,df,outputte,labeltte,indexte,batch_size=32):
  """It does the same as performances functon, but just for a specific feature. 

  Args:
  ----------
  feature : String
        any of the feature of the df
  df : pandas.core.frame.DataFrame
        whole DataFrame used for this experiment. It contains train, validation and test data
  outputte : list of torch.Tensor
        list of all the torch.Tensors we have in output from the model
  labeltte : list of torch.Tensor
        list of all the torch.Tensors that are the label for the model
  indexte : list of torch.Tensor
        list of all the torch.Tensors that we save to match the same window in df and the DataLoader, by the use of the index
  batch_size int (Default=32)
        Size of the batch used during the training of the model.

  Returns:
  ----------
  feature_of_all : list
        list of all values of "feature" for the traces in the test set
  feature_of_wrong_pre : list
        list of values of "feature" for the traces in the test set where the label is pre, but the model predict class="post"
  feature_of_wrong_post : list
        list of values of "feature" for the traces in the test set where the label is post, but the model predict class="pre"
  """

  feature_all=[]
  feature_of_wrong_pre=[]
  feature_of_wrong_post=[]
  if feature not in df.columns:
      print("Feature ",feature," not in df columns")

  for w in range(len(outputte)):
    for b in range(0,batch_size):
      #tot=tot+1
      feature_all.append(df[feature].iloc[indexte[w][b].item()])
      if labeltte[w][b][0]==0: #post
        if not torch.all(to_one_hot(outputte[w][b].cpu()).eq(labeltte[w][b].cpu())):
          feature_of_wrong_post.append(df[feature].iloc[indexte[w][b].item()])
      else: #pre
        if not torch.all(to_one_hot(outputte[w][b].cpu()).eq(labeltte[w][b].cpu())):
          feature_of_wrong_pre.append(df[feature].iloc[indexte[w][b].item()])
  return feature_all, feature_of_wrong_pre, feature_of_wrong_post


def create_output_testDF(df, outputte, labeltte, indexte, batch_size=32):
  """It does the same as performances functon, but just for a specific feature. 

  Args:
  ----------
  df : pandas.core.frame.DataFrame
        whole DataFrame used for this experiment. It contains train, validation and test data
  outputte : list of torch.Tensor
        list of all the torch.Tensors we have in output from the model
  labeltte : list of torch.Tensor
        list of all the torch.Tensors that are the label for the model
  indexte : list of torch.Tensor
        list of all the torch.Tensors that we save to match the same window in df and the DataLoader, by the use of the index
  batch_size int (Default=32)
        Size of the batch used during the training of the model.

  Returns:
  ----------
  whole_df : pandas.core.frame.DataFrame
        same of df but with added columns output and binary_output
  """

  out_df = pd.DataFrame(columns=['trace_name','output', 'binary_output'])
  c=0
  for w in range(len(outputte)):
    for b in range(0,batch_size):
      one_hot=to_one_hot(outputte[w][b].cpu())
      one_hot = [int(x) for x in one_hot]
      dict = {'trace_name': df['trace_name'].iloc[indexte[w][b].item()], 'output': [outputte[w][b].cpu().numpy()], 'binary_output': [one_hot]}
      out_dict=pd.DataFrame.from_dict(dict)
      out_df = pd.concat([out_df, out_dict], ignore_index=True)
      c=c+1
  whole_df = pd.merge(out_df, df, on=['trace_name'])
  return whole_df


from sklearn.preprocessing import MinMaxScaler

def PlotEventsWithStations(dataToProcess,mdl,output_testDF,save_img_out=False,img_name=""):
  """ This plots events (all and wrong classified) together with stations and main events.
  Args:
  ----------
  dataToProcess : String
        "mini" or "all" or single station (e.g. "T1213" or "NRCA"...)
  mdl : String
        adopted model to be written in plot title. "FCN_saliency" or "FCN" or "CNN" or "CNN_saliency"
  output_testDF : pandas.core.frame.DataFrame
        df that includes outputs from the ML model
  """
  # Importing stations
  stations = {'name': ['NRCA',  'MMO1', 'MC2', 'FDMO', 'T1212', 'T1213', 'T1214', 'T1216', 'T1244'],
            'lat': [42.83355, 42.899333, 42.911418, 43.0365, 42.751556, 42.724918, 42.759537, 42.890667, 42.75697],
            'lon': [13.11427, 13.326833, 13.188976, 13.0873, 13.044636, 13.125775, 13.208697, 13.019000, 13.29779],
            'alt': [927, 957, 2, 550, 869, 860, 1490, 620, 950]}
  
  stations = pd.DataFrame(data=stations)

  # Importing locations
  locations = {'name': ['Visso', 'Norcia', 'Amatrice', 'Accumoli', 'Campotosto'],
        'lat': [42.9303, 42.7942, 42.628016, 42.694592, 42.553978],
        'lon': [13.0885, 13.0964, 13.292479, 13.245461, 13.370281]}

  locations = pd.DataFrame(data=locations)

  #Greater events
  grev = pd.read_csv('./raw_data/greater_events.csv')

  # Principal events
  prev = grev
  prev = prev.reset_index(drop=True)
  prev = prev.drop([1, 2, 5, 6, 7, 8]).reset_index(drop=True)

  # Norcia mainshock
  mainev_lon, mainev_lat, mainev_dth  = 13.12131, 42.83603, 6.104
  
  correct_df = pd.DataFrame(columns=['trace_name', 'source_latitude', 'source_longitude', 'output'])
  wrong_pre_df = pd.DataFrame(columns=['trace_name', 'source_latitude', 'source_longitude', 'output'])
  wrong_post_df = pd.DataFrame(columns=['trace_name', 'source_latitude', 'source_longitude', 'output'])
  for index, row in output_testDF.iterrows():
    if row['label']==row['binary_output']: #correct
        correct_df.loc[len(correct_df.index)] = [row['trace_name'],row['source_latitude'],row['source_longitude'],row['output']]
    else: #wrong
        if row['label'][0]==1: #pre classified as post
            wrong_pre_df.loc[len(wrong_pre_df.index)] = [row['trace_name'],row['source_latitude'],row['source_longitude'],row['output'][1]]
        else: #post classified as pre
            wrong_post_df.loc[len(wrong_post_df.index)] = [row['trace_name'],row['source_latitude'],row['source_longitude'],row['output'][0]]
  scaler = MinMaxScaler(feature_range=(0,10))
  if not wrong_pre_df.empty:
    wrong_pre_df[["output"]] = scaler.fit_transform(wrong_pre_df[["output"]])
  if not wrong_post_df.empty:
    wrong_post_df[["output"]] = scaler.fit_transform(wrong_post_df[["output"]])
  f1 = plt.figure(figsize=(10, 10))
  plt.title('Location of EQ for station'+dataToProcess+' and model '+mdl, fontsize=14)
  plt.scatter(mainev_lon, mainev_lat, c='red', marker='*', s=400, label='Principal events')
  plt.scatter(grev['lon'], grev['lat'], c='red', marker='*', s=100, alpha=0.5)
  plt.scatter(stations['lon'], stations['lat'], c='k', marker='^', s=50, label='Stations')
  plt.scatter(locations[0:2]['lon'], locations[0:2]['lat'], c='k', marker='s', s=10)
  cor=plt.scatter(x=correct_df['source_longitude'], y=correct_df['source_latitude'], color="g", s=4, alpha=1, label='correct')
  if wrong_pre_df.empty:
    print('DataFrame wrong_pre_df is empty!')
  else:
    w_pre=plt.scatter(x=wrong_pre_df['source_longitude'], y=wrong_pre_df['source_latitude'], c=wrong_pre_df['output'], cmap='autumn_r', s=4, alpha=1, label='wrong pre')#
  if wrong_post_df.empty:
    print('DataFrame wrong_post_df is empty!')
  else:
    w_post=plt.scatter(x=wrong_post_df['source_longitude'], y=wrong_post_df['source_latitude'], c=wrong_post_df['output'], cmap='cool', s=4, alpha=1, label='wrong post')#
  for i in range(0, len(stations)):
    plt.text(stations['lon'][i], stations['lat'][i], stations['name'][i], color='k')
  for i in range(0, len(locations[0:2])):
    plt.text(locations['lon'][i], locations['lat'][i], locations['name'][i], color='k')
  plt.xlabel('Longitude [°]', fontsize=12)
  plt.ylabel('Latitude [°]', fontsize=12)
  #plt.xlim(330,370)
  plt.ylim(42.7,43.05)
  plt.gca().set_aspect('equal')
  plt.grid(True, alpha=0.1)
  if not wrong_pre_df.empty:
    plt.colorbar(w_pre,fraction=0.0422, pad=0.04)
  if not wrong_post_df.empty:
    plt.colorbar(w_post,fraction=0.046, pad=0.04)
  lgnd = plt.legend(loc='upper right')
  if wrong_pre_df.empty or wrong_post_df.empty:
    range_end=4
  else:
    range_end=5
  for i in range(0, range_end):
    lgnd.legendHandles[i]._sizes = [70]
  if save_img_out:
      plt.savefig(img_name,bbox_inches='tight')


def PlotHistOfFeature(output_testDF,dataToProcess, mdl, seed, num_classes,feature, hist_bins=20, plot_together=False, plot_percentage=False, save_img_out=False,path="", TTF_measure="sec"):  
    """ This plots three histograms for the chosen feature, one for all traces and two for wrongly classified ones
    Args:
    ----------
    output_testDF : pandas.core.frame.DataFrame
        df that includes outputs from the ML model
    dataToProcess : String
        "mini" or "all" or single station (e.g. "T1213" or "NRCA"...)
    mdl : String
        adopted model to be written in plot title. "FCN_saliency" or "FCN" or "CNN" or "CNN_saliency"
    seed : int
        Seed for all the methods
    num_classes : int
        number of total classes we eant to split the df (both pre and post)
    feature : String
        column (feature) of output_testDF to be analyzed
    hist_bins : int
        number of bins in the histograms
    plot_together : Bool (Default=False)
        If true it plots the hist for pre and post in the same panel
    plot_percentage : Bool (Default=False)
        If true it plots the variation in % of classified samples
    save_img_out : Bool (Default=False)
        If true it saves the plots in output
    path : String
        Path where to save the data
    TTF_measure : String (Default="sec")
        measure for Time To Failure. It is implemented for "sec" or "days"
    """  

    if TTF_measure=="days":
        #this is the converting measure to change from seconds to days
        div=86400
        measure_unit=" (days)"
    elif TTF_measure=="sec":
        #this is the converting measure to change from seconds to days
        div=1
        measure_unit=" (sec)"
    else: 
        div=1
        measure_unit=""
    plt.figure(figsize=(8,5))
    n_all, bins_all, patches_all = plt.hist(output_testDF[feature]/div, bins=hist_bins, color='grey', edgecolor='black')
    plt.title(feature+' of all EQ in test set')
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Number of traces', fontsize=12)
    plt.grid(True, axis='y', alpha=0.1, zorder=-10)
    if save_img_out:
        img_name=path+"\\images\\"+dataToProcess+"_"+feature+"HistAll_"+mdl+"_seed"+str(seed)+"_classes"+str(num_classes)+".jpg"
        plt.savefig(img_name,bbox_inches='tight')
    plt.show()
    
    feature_of_actual_pre=output_testDF.copy() 
    feature_of_actual_pre['label'] = feature_of_actual_pre['label'].apply(lambda x: str(x))
    feature_of_actual_pre=feature_of_actual_pre[feature_of_actual_pre['label'] == '[1, 0]']
    n_actual_pre, _, _=plt.hist(feature_of_actual_pre[feature]/div, bins=hist_bins, range=(bins_all.min(), bins_all.max()), label='actual pre')
    plt.close()

    feature_of_actual_post=output_testDF.copy() 
    feature_of_actual_post['label'] = feature_of_actual_post['label'].apply(lambda x: str(x))
    feature_of_actual_post=feature_of_actual_post[feature_of_actual_post['label'] == '[0, 1]']
    n_actual_post, _, _=plt.hist(feature_of_actual_post[feature]/div, bins=hist_bins, range=(bins_all.min(), bins_all.max()), label='actual post')
    plt.close()

    # correct_pre
    plt.figure(figsize=(8,5))
    feature_of_correct_pre=output_testDF.copy() 
    feature_of_correct_pre['label'] = feature_of_correct_pre['label'].apply(lambda x: str(x))
    feature_of_correct_pre['binary_output'] = feature_of_correct_pre['binary_output'].apply(lambda x: str(x))
    feature_of_correct_pre=feature_of_correct_pre[(feature_of_correct_pre['label'] == '[1, 0]') & (feature_of_correct_pre['binary_output'] == '[1, 0]')]
    n_correct_pre, bins_correct_pre, patches_correct_pre=plt.hist(feature_of_correct_pre[feature]/div, bins=hist_bins, range=(bins_all.min(), bins_all.max()), color='yellow', edgecolor='darkgoldenrod', label='correct pre')
    rects_correct_pre = patches_correct_pre.patches
    labels_correct_pre = []
    for i in range(0,hist_bins):
        if n_actual_pre[i]==0:
            labels_correct_pre.append(np.nan)
        else:
            labels_correct_pre.append(int(n_correct_pre[i]/n_actual_pre[i]*100))
    labels_correct_pre_for_labels = [] # this is to make possible to plot together with the correct labels, differentiating nan and 0
    for i in range(0,hist_bins):
        if n_all[i]==0:
            labels_correct_pre_for_labels.append(np.nan)
        else:
            labels_correct_pre_for_labels.append(int(n_correct_pre[i]/n_all[i]*100))
    if not plot_together:
        labels_correct_pre_str=[str(x) if np.isnan(x) else str(x)+'%' for x in labels_correct_pre]
        for rect, label in zip(rects_correct_pre,labels_correct_pre_str):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height+0.01, label, ha='center', va='bottom')
        plt.title(feature+' of pre correctly classified (% of correct in bin)')
        plt.xlabel(feature+measure_unit, fontsize=12)
        plt.ylabel('Number of traces', fontsize=12)
        plt.grid(True, axis='y', alpha=0.1, zorder=-10)
        if save_img_out:
            img_name=path+"\\images\\"+dataToProcess+"_"+feature+"HistCorrectPre_"+mdl+"_seed"+str(seed)+"_classes"+str(num_classes)+".jpg"
            plt.savefig(img_name,bbox_inches='tight')
        plt.show()

    # correct_post
    if not plot_together:
      plt.figure(figsize=(8,5))
    feature_of_correct_post=output_testDF.copy() 
    feature_of_correct_post['label'] = feature_of_correct_post['label'].apply(lambda x: str(x))
    feature_of_correct_post['binary_output'] = feature_of_correct_post['binary_output'].apply(lambda x: str(x))
    feature_of_correct_post=feature_of_correct_post[(feature_of_correct_post['label'] == '[0, 1]') & (feature_of_correct_post['binary_output'] == '[0, 1]')]
    n_correct_post, bins_correct_post, patches_correct_post=plt.hist(feature_of_correct_post[feature]/div, bins=hist_bins, range=(bins_all.min(), bins_all.max()),color='lime', edgecolor='darkgreen', label='correct post')
    rects_correct_post = patches_correct_post.patches
    labels_correct_post = []
    for i in range(0,hist_bins):
        if n_actual_post[i]==0:
            labels_correct_post.append(np.nan)
        else:
            labels_correct_post.append(int(n_correct_post[i]/n_actual_post[i]*100))
    
    labels_correct_post_for_labels = [] # this is to make possible to plot together with the correct labels, differentiating nan and 0
    for i in range(0,hist_bins):
        if n_all[i]==0:
            labels_correct_post_for_labels.append(np.nan)
        else:
            labels_correct_post_for_labels.append(int(n_correct_post[i]/n_all[i]*100))
    if not plot_together:
        labels_correct_post_str=[str(x) if np.isnan(x) else str(x)+'%' for x in labels_correct_post]
        for rect, label in zip(rects_correct_post ,labels_correct_post_str):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height+0.01, label, ha='center', va='bottom')
        plt.title(feature+' of post correctly classified (% of correct in bin)')
        plt.xlabel(feature+measure_unit, fontsize=12)
        plt.ylabel('Number of traces', fontsize=12)
        plt.grid(True, axis='y', alpha=0.1, zorder=-10)
        if save_img_out:
            img_name=path+"\\images\\"+dataToProcess+"_"+feature+"HistCorrectPost_"+mdl+"_seed"+str(seed)+"_classes"+str(num_classes)+".jpg"
            plt.savefig(img_name,bbox_inches='tight')
        plt.show()

    labels=np.array(0)
    # if plot_together:
    labels=np.array([np.array(labels_correct_pre_for_labels), np.array(labels_correct_post_for_labels)])
    labels=np.sum(labels, axis=0)
    if plot_together:
        labels_all=[str(x) if np.isnan(x) else str(x)+'%'  for x in labels]
        for rect_pre,rect_post, label in zip(rects_correct_pre,rects_correct_post,labels_all):
            height = max(rect_pre.get_height(),rect_post.get_height())
            plt.text(rect_pre.get_x() + rect_pre.get_width() / 2, height+0.01, label, ha='center', va='bottom')
        plt.title(feature+' of correctly classified EQ in test set (% of correct in bin)')
        plt.xlabel(feature+measure_unit, fontsize=12)
        plt.ylabel('Number of traces', fontsize=12)
        plt.legend(loc='lower right')
        plt.grid(True, axis='y', alpha=0.1, zorder=-10)
        if save_img_out:
            img_name=path+"\\images\\"+dataToProcess+"_"+feature+"HistCorrect_"+mdl+"_seed"+str(seed)+"_classes"+str(num_classes)+".jpg"
            plt.savefig(img_name,bbox_inches='tight')
        plt.show()
    
    if plot_percentage:
      if not np.all(labels==0):
        plt.figure(figsize=(8,5))
        plt.plot(bins_all[:-1],labels, color='black', marker='o')
        plt.title('% of correctly classified EQ in test set')
        plt.xlabel(feature+measure_unit, fontsize=12)
        plt.ylabel('Percentage', fontsize=12)
        plt.grid(True, axis='y', alpha=0.1, zorder=-10)
        plt.ylim(-5, 105)
        if save_img_out:
            img_name=path+"\\images\\"+dataToProcess+"_"+feature+"HistPercentageCorrect_"+mdl+"_seed"+str(seed)+"_classes"+str(num_classes)+".jpg"
            plt.savefig(img_name,bbox_inches='tight')
        plt.show()
    
    # wrong_pre
    plt.figure(figsize=(8,5))
    feature_of_wrong_pre=output_testDF.copy() 
    feature_of_wrong_pre['label'] = feature_of_wrong_pre['label'].apply(lambda x: str(x))
    feature_of_wrong_pre['binary_output'] = feature_of_wrong_pre['binary_output'].apply(lambda x: str(x))
    feature_of_wrong_pre=feature_of_wrong_pre[(feature_of_wrong_pre['label'] == '[1, 0]') & (feature_of_wrong_pre['binary_output'] == '[0, 1]')]
    n_wrong_pre, bins_wrong_pre, patches_wrong_pre=plt.hist(feature_of_wrong_pre[feature]/div, bins=hist_bins, range=(bins_all.min(), bins_all.max()), color='fuchsia', edgecolor='purple', label='wrong pre')
    rects_wrong_pre = patches_wrong_pre.patches
    labels_wrong_pre = []
    for i in range(0,hist_bins):
        if n_actual_pre[i]==0:
            labels_wrong_pre.append(np.nan)
        else:
            labels_wrong_pre.append(int(n_wrong_pre[i]/n_actual_pre[i]*100))
    
    labels_wrong_pre_for_labels = [] # this is to make possible to plot together with the correct labels, differentiating nan and 0
    for i in range(0,hist_bins):
        if n_all[i]==0:
            labels_wrong_pre_for_labels.append(np.nan)
        else:
            labels_wrong_pre_for_labels.append(int(n_wrong_pre[i]/n_all[i]*100))     
    if not plot_together:   
        labels_wrong_pre_str=[str(x) if np.isnan(x) else str(x)+'%' for x in labels_wrong_pre]
        for rect, label in zip(rects_wrong_pre,labels_wrong_pre_str):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height+0.01, label, ha='center', va='bottom')
        plt.title(feature+' of pre classified as post (% of wrong in bin)')
        plt.xlabel(feature+measure_unit, fontsize=12)
        plt.ylabel('Number of traces', fontsize=12)
        plt.grid(True, axis='y', alpha=0.1, zorder=-10)
        if save_img_out:
            img_name=path+"\\images\\"+dataToProcess+"_"+feature+"HistPreAsPost_"+mdl+"_seed"+str(seed)+"_classes"+str(num_classes)+".jpg"
            plt.savefig(img_name,bbox_inches='tight')
        plt.show()
    
    # wrong_post
    if not plot_together:
      plt.figure(figsize=(8,5))
    feature_of_wrong_post=output_testDF.copy() 
    feature_of_wrong_post['label'] = feature_of_wrong_post['label'].apply(lambda x: str(x))
    feature_of_wrong_post['binary_output'] = feature_of_wrong_post['binary_output'].apply(lambda x: str(x))
    feature_of_wrong_post=feature_of_wrong_post[(feature_of_wrong_post['label'] == '[0, 1]') & (feature_of_wrong_post['binary_output'] == '[1, 0]')]
    n_wrong_post, bins_wrong_post, patches_wrong_post=plt.hist(feature_of_wrong_post[feature]/div, bins=hist_bins, range=(bins_all.min(), bins_all.max()), color='blue', edgecolor='navy', label='wrong post')
    rects_wrong_post = patches_wrong_post.patches
    labels_wrong_post = []
    for i in range(0,hist_bins):
        if n_actual_post[i]==0:
            labels_wrong_post.append(np.nan)
        else:
            labels_wrong_post.append(int(n_wrong_post[i]/n_actual_post[i]*100))
    labels_wrong_post_for_labels = [] # this is to make possible to plot together with the correct labels, differentiating nan and 0
    for i in range(0,hist_bins):
        if n_all[i]==0:
            labels_wrong_post_for_labels.append(np.nan)
        else:
            labels_wrong_post_for_labels.append(int(n_wrong_post[i]/n_all[i]*100))    
    if not plot_together:    
        labels_wrong_post_str=[str(x) if np.isnan(x) else str(x)+'%' for x in labels_wrong_post]
        for rect, label in zip(rects_wrong_post ,labels_wrong_post_str):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height+0.01, label, ha='center', va='bottom')
        plt.title(feature+' of post classified as pre (% of wrong in bin)')
        plt.xlabel(feature+measure_unit, fontsize=12)
        plt.ylabel('Number of traces', fontsize=12)
        plt.grid(True, axis='y', alpha=0.1, zorder=-10)
        if save_img_out:
            img_name=path+"\\images\\"+dataToProcess+"_"+feature+"HistPostAsPre_"+mdl+"_seed"+str(seed)+"_classes"+str(num_classes)+".jpg"
            plt.savefig(img_name,bbox_inches='tight')
        plt.show()
    
    labels=np.array(0)
    #if plot_together:
    labels=np.array([np.array(labels_wrong_pre_for_labels), np.array(labels_wrong_post_for_labels)])
    labels=np.sum(labels, axis=0)
    if plot_together:
        labels_all=[str(x) if np.isnan(x) else str(x)+'%' for x in labels]
        for rect_pre,rect_post, label in zip(rects_wrong_pre,rects_wrong_post,labels_all):
            height = max(rect_pre.get_height(),rect_post.get_height())
            plt.text(rect_pre.get_x() + rect_pre.get_width() / 2, height+0.01, label, ha='center', va='bottom')
        plt.title(feature+' of wrongly classified EQ in test set (% of wrong in bin)')
        plt.xlabel(feature+measure_unit, fontsize=12)
        plt.ylabel('Number of traces', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, axis='y', alpha=0.1, zorder=-10)
        if save_img_out:
            img_name=path+"\\images\\"+dataToProcess+"_"+feature+"HistWrong_"+mdl+"_seed"+str(seed)+"_classes"+str(num_classes)+".jpg"
            plt.savefig(img_name,bbox_inches='tight')
        plt.show()
    
    if plot_percentage:
      if not np.all(labels==0):
        plt.figure(figsize=(8,5))
        plt.plot(bins_all[:-1],labels, color='black', marker='o')
        plt.title('% of wrongly classified EQ in test set')
        plt.xlabel(feature+measure_unit, fontsize=12)
        plt.ylabel('Percentage', fontsize=12)
        plt.grid(True, axis='y', alpha=0.1, zorder=-10)
        plt.ylim(-5, 105)
        if save_img_out:
            img_name=path+"\\images\\"+dataToProcess+"_"+feature+"HistPercentageWrong_"+mdl+"_seed"+str(seed)+"_classes"+str(num_classes)+".jpg"
            plt.savefig(img_name,bbox_inches='tight')
        plt.show()



###### FUNCTIONS FOR SALIENCY MAPS VISUALIIZATION #######

def NormalizeData(data):
  """It takes an array in as an input and normalizes its values between 0 and 1.
     It then returns an output array with the same dimensions as the input.

  Args:
  ----------
  data : np.array
        data to normalize

  Returns:
  ----------
  normalized data

  """
  return (data - np.min(data)) / (np.max(data) - np.min(data))


def plot_with_saliency(X_inp,saliency_inp,lab,threshold=0.4, show_spectrogram=False):
  """It takes an array in as an input and normalizes its values between 0 and 1.
     It then returns an output array with the same dimensions as the input.
     The bigger and darker the points, the more important are for the model in the classification process.

  Args:
  ----------
  X_inp : array
        input data to visulize
  saliency_inp : np.array
        saliency map corresponding to input data
  lab : string
        channel we are plotting ("E" or "N" or "Z"). This is considered as label in plt.plot function.
  threshold : float 0<=threshold<=1 (Default=0.4)
        Threshold from where to start visualizing points of the saliency map. This is to avoid being distracting
        by irrelevant points
  show_spectrogram : Bool (Default=False)
        If true it plots the spectrogram corresponding to the seismogram

  """
  plt.plot(X_inp,alpha=0.7, label=[lab])
  saliency_norm=NormalizeData(saliency_inp)
  saliency_norm[saliency_norm<threshold] = 0 # I remove the point that are not so important
  inp_size=len(X_inp)
  for j in range(0,inp_size-1,2):
    plt.plot([j,j+1],[X_inp[j],X_inp[j+1]],'.-',c='r',alpha=saliency_norm[j],markersize=saliency_norm[j]*15)#saliency_inp[j]*0.5)
  plt.xlabel('data points (#)')
  plt.ylabel('waveform')
  plt.legend(loc='upper left')
  plt.show()
  if show_spectrogram:
      #wZT=df_postStation.loc[df_postStation['associato_ev'] == i]["E_channel"].values[0]
      im=spectrogram(data=X_inp, samp_rate=1/0.01,dbscale=False, log=False,cmap='jet')#, axes=ax[0,2]
      #plt.show()

def compute_saliency_maps(X, y, model, device, mdl, inp_size, saliency_target="ALL", s_max=False):
  """
  It compute a class saliency map using the model for signal X and labels y.

  Args:
  ----------
  X : array
        input signal
  y : array
        Labels for X
  model : torch.model
        A pretrained model that will be used to compute the saliency map.
        Note that the model to use is "FCN_for_saliency", otherwise there will be errors.
  mdl : String
        adopted model to be written in plot title. "FCN_saliency" or "CNN_saliency"
  inp_size : int
        length of each channel in one input sequence
  saliency_target : string (Default="ALL")
        IF "ALL" the saliency maps shows relevant points for predicting the correct class.
        IF "pre" the saliency maps shows relevant points for predicting pre.
        IF "post" the saliency maps shows relevant points for predicting post.
  s_max : Bool (Default=False)
        If depend on the settings of the model
  Returns:
  ----------
  saliency: A Tensor giving the saliency maps for the input
  out: output of the model

  """

  model.eval() # Make sure the model is in "test" mode
  X.requires_grad_() # Make input tensor require gradient

  saliency = None
  ##############################################################################
  # TODO: Implement this function. Perform a forward and backward pass through #
  # the model to compute the gradient of the correct class score with respect  #
  # to each input image. You first want to compute the loss over the correct   #
  # scores (we'll combine losses across a batch by summing), and then compute  #
  # the gradients with a backward pass.                                        #
  ##############################################################################

  # Forward pass
  if mdl=="FCN_saliency":
      b_size=model.state_dict()["layer_3.bias"].size()[0]
      in_size=inp_size
      scores = model(X,batch_size=b_size, steps_in=in_size)
  elif mdl=="CNN_saliency":
      b_size=model.state_dict()["conv1.bias"].size()[0]
      in_size=inp_size
      scores = model(X,batch_size=b_size, steps_in=in_size, softmax=s_max)[0]
  out=scores.clone()
  # Backward pass
  if saliency_target=="ALL": # focus for predicting correct class
    scores = scores.gather(-1, y)[:,1] # Correct class scores
    scores.backward(torch.ones(scores.size()).to(device))
  elif saliency_target=="post": # focus for predicting post
    scores[:,0].backward(torch.ones(scores[:,0].size()))
  elif saliency_target=="pre": # focus for predicting pre
    scores[:,1].backward(torch.ones(scores[:,1].size()))
  else:
    print("saliency_target not correct. Chose ALL or pre or post")

  saliency = X.grad
  saliency = saliency.abs()
  # Convert 3d to 1d
  #saliency, _= torch.max(saliency, dim=2)
  return saliency, out


def show_saliency_maps(X, y, model, device, mdl, inp_size, saliency_target="ALL", s_max=False,show_spectrogram=False):
  """
  It shows the saliency map using the model for signal X and labels y.
  It also prints performances

  Args:
  ----------
  X : array
        input signal
  y : array
        Labels for X
  model : torch.model
        A pretrained model that will be used to compute the saliency map.
        Note that the model to use is "FCN_for_saliency", otherwise there will be errors.
  mdl : String
        adopted model to be written in plot title. "FCN_saliency" or "FCN" or "CNN" or "CNN_saliency"
  inp_size : int
        length of each channel in one input sequence
  saliency_target : string (Default="ALL")
        IF "ALL" the saliency maps shows relevant points for predicting the correct class.
        IF "pre" the saliency maps shows relevant points for predicting pre.
        IF "post" the saliency maps shows relevant points for predicting post.
  s_max : Bool (Default=False)
        If depend on the settings of the model
  show_spectrogram : Bool (Default=False)
        If true it plots the spectrogram corresponding to the seismogram
  """
  tot=0
  wrong=0
  tot_pre=0
  wrong_pre=0
  tot_post=0
  wrong_post=0
  saliency,out = compute_saliency_maps(X, y, model, device, mdl, inp_size,saliency_target=saliency_target, s_max=s_max)

  # Convert the saliency map from Torch Tensor to numpy array and shows input signal and saliency maps together.
  saliency = saliency.cpu().numpy()
  X = X.cpu().detach().numpy()
  N = X.shape[0]
  threshold=0.4
  for i in range(0,N):
    plot_with_saliency(X[i][:,0],saliency[i][:,0],'E', threshold,show_spectrogram=show_spectrogram)
    plot_with_saliency(X[i][:,1],saliency[i][:,1],'N', threshold,show_spectrogram=show_spectrogram)
    plot_with_saliency(X[i][:,2],saliency[i][:,2],'Z', threshold,show_spectrogram=show_spectrogram)
    print("output=",out[i])
    print("label=",y[i])
    print("binary output=",to_one_hot(out[i].cpu().detach().numpy()))
    tot=tot+1
    if not torch.all(to_one_hot(out[i].cpu().detach().numpy()).type(torch.IntTensor).eq(y[i].cpu())):
      wrong=wrong+1
    if y[i][0]==1: #post
      tot_post=tot_post+1
      if not torch.all(to_one_hot(out[i].cpu().detach().numpy()).type(torch.IntTensor).eq(y[i].cpu())):
        wrong_post=wrong_post+1
    else: #pre
      tot_pre=tot_pre+1
      if not torch.all(to_one_hot(out[i].cpu().detach().numpy()).type(torch.IntTensor).eq(y[i].cpu())):
        wrong_pre=wrong_pre+1
  print("tot: ", tot," wrong: ",wrong)
  print("percentage of correctly classified: ",((tot-wrong)/tot*100),"%")
  print("tot pre: ", tot_pre," wrong pre: ",wrong_pre)
  print("percentage of pre correctly classified: ",((tot_pre-wrong_pre)/tot_pre*100),"%")
  print("tot post: ", tot_post," wrong post: ",wrong_post)
  print("percentage of post correctly classified: ",((tot_post-wrong_post)/tot_post*100),"%")



###### FUNCTIONS FOR SECTION PLOTS #######


def section_plots(dataframe, station, save=False):
      # Importing Tan dataset
      tan = pd.read_csv('./sections_data/Amatrice_CAT5.v20210325', delimiter='\s+', header=19)

      tan.columns = ['year', 'month', 'day', 'hour', 'min', 'secmsec', 'lat', 'lon', 'depth', 'EH1', 'EH2', 'AZ', 'EZ', 'Ml', 'Mw', 'ID']
      tan = tan.drop(['EH1', 'EH2', 'AZ', 'EZ'], axis=1)

      df = pd.DataFrame({'year': tan['year'],
                        'month': tan['month'],
                        'day': tan['day'],
                        'hour': tan['hour'],
                        'minute': tan['min'], 
                        'second': tan['secmsec']
      })
      tan['time'] = pd.to_datetime(df, format='%d%m%y %H:%M:%S')
      tan = tan.drop(['year', 'month', 'day', 'hour', 'min', 'secmsec'], axis=1)

      tan = tan.reset_index(drop=True)

      # Importing stations
      stations = {'name': ['NRCA',  'MMO1', 'MC2', 'FDMO', 'T1212', 'T1213', 'T1214', 'T1216', 'T1244'],
                  'lat': [42.83355, 42.899333, 42.911418, 43.0365, 42.751556, 42.724918, 42.759537, 42.890667, 42.75697],
                  'lon': [13.11427, 13.326833, 13.188976, 13.0873, 13.044636, 13.125775, 13.208697, 13.019000, 13.29779],
                  'alt': [927, 957, 2, 550, 869, 860, 1490, 620, 950]}

      stations = pd.DataFrame(data=stations)

      # Importing locations
      locations = {'name': ['Visso', 'Norcia', 'Amatrice', 'Accumoli', 'Campotosto'],
            'lat': [42.9303, 42.7942, 42.628016, 42.694592, 42.553978],
            'lon': [13.0885, 13.0964, 13.292479, 13.245461, 13.370281]}

      locations = pd.DataFrame(data=locations)

      # Importing selected events
      catalog_pre = pd.read_csv('./sections_data/catalog_pre.csv')
      catalog_post = pd.read_csv('./sections_data/catalog_post.csv')

      # Greater events
      grev = pd.read_csv('./sections_data/greater_events.csv')

      # Principal events
      prev = grev
      prev = prev.reset_index(drop=True)
      prev = prev.drop([1, 2, 5, 6, 7, 8]).reset_index(drop=True)

      # Wrong events
      #output_catalog = pd.read_pickle(f'./sections_data/{dataframe}')
      output_catalog = dataframe

      wrev = output_catalog.loc[output_catalog.binary_output != output_catalog.label]
      wrev = wrev.reset_index(drop=True)

      wrev_pre = wrev.loc[wrev.trace_start_time < '2016-10-31 07:41:00']
      wrev_pre = wrev_pre.reset_index(drop=True)

      wrev_post = wrev.loc[wrev.trace_start_time > '2016-10-31 07:41:00']
      wrev_post = wrev_post.reset_index(drop=True)
      
      import pyproj
      utm33 = pyproj.Proj(proj='utm', zone=33, ellps='WGS84', datum='WGS84', units='m')

      # Events
      utmx_tan, utmy_tan = utm33(np.array(tan['lon']), np.array(tan['lat']))
      utmx_pre, utmy_pre = utm33(np.array(catalog_pre['lon']), np.array(catalog_pre['lat']))
      utmx_post, utmy_post = utm33(np.array(catalog_post['lon']), np.array(catalog_post['lat']))

      # Stations and locations
      utmx_stat, utmy_stat = utm33(np.array(stations['lon']), np.array(stations['lat']))
      utmx_loc, utmy_loc = utm33(np.array(locations['lon']), np.array(locations['lat']))

      # Greater events
      utmx_grev, utmy_grev = utm33(np.array(grev['lon']), np.array(grev['lat']))
      utmx_prev, utmy_prev = utm33(np.array(prev['lon']), np.array(prev['lat']))

      # Main events
      amatrice_lon, amatrice_lat, amatrice_depth = 13.2428, 42.7145, 3.276
      utmx_amatrice, utmy_amatrice = utm33(amatrice_lon, amatrice_lat)
      amatrice = np.array([utmx_amatrice/1000, utmy_amatrice/1000, -amatrice_depth])

      amatrice_af_lon, amatrice_af_lat, amatrice_af_depth = 13.15885, 42.80178, 4.113
      utmx_amatrice_af, utmy_amatrice_af = utm33(amatrice_af_lon, amatrice_af_lat)
      amatrice_af = np.array([utmx_amatrice_af/1000, utmy_amatrice_af/1000, -amatrice_af_depth])

      visso_lon, visso_lat, visso_depth = 13.1394, 42.9109, 2.605
      utmx_visso, utmy_visso = utm33(visso_lon, visso_lat)
      visso = np.array([utmx_visso/1000, utmy_visso/1000, -visso_depth])

      visso2_lon, visso2_lat, visso2_depth = 13.12915	, 42.88725, 1.672
      utmx_visso2, utmy_visso2 = utm33(visso2_lon, visso2_lat)
      visso2 = np.array([utmx_visso2/1000, utmy_visso2/1000, -visso2_depth])

      norcia_lon, norcia_lat, norcia_depth = 13.12131, 42.83603, 6.104
      utmx_norcia, utmy_norcia = utm33(norcia_lon, norcia_lat)
      norcia = np.array([utmx_norcia/1000, utmy_norcia/1000, -norcia_depth])

      # Wrong events
      utmx_wrev_pre, utmy_wrev_pre = utm33(np.array(wrev_pre['source_longitude']), np.array(wrev_pre['source_latitude']))
      utmx_wrev_post, utmy_wrev_post = utm33(np.array(wrev_post['source_longitude']), np.array(wrev_post['source_latitude']))

      def distance_point_from_plane(x, y, z, normal, origin):
            d = -normal[0]*origin[0]-normal[1]*origin[1]-normal[2]*origin[2]
            dist = np.abs(normal[0]*x+normal[1]*y+normal[2]*z+d)
            dist = dist/np.sqrt(normal[0]**2+normal[1]**2+normal[2]**2)
            return dist

      # Direction of the section to plot
      normal_tostrike = 155 - 90
      normal_ref = [np.cos(normal_tostrike*np.pi/180), -np.sin(normal_tostrike*np.pi/180), 0]

      # Distance from a plane
      dist = distance_point_from_plane(utmx_tan/1000,utmy_tan/1000, -tan['depth'], normal_ref, amatrice_af)

      # Distance from a plane < x 
      resultdist1m = np.where(dist <1)

      # Distance on a section
      X_onsection1m =+ (utmy_tan[resultdist1m]/1000-amatrice_af[1])*normal_ref[0]-(utmx_tan[resultdist1m]/1000-amatrice_af[0])*normal_ref[1]

      # Distance between each section
      d = np.arange(-26, 18, 2)   # Numero di punti
      x, y = norcia[0], norcia[1] # Coordinate del perno
      theta, theta_2 = 155, 335   # Strike

      def point_position(x, y, d, theta):
            # Trova le coordinate dei punti equidistanti 2 km da Norcia
            theta_rad = np.pi/2 - np.radians(theta)
            return x + d*np.cos(theta_rad), y + d*np.sin(theta_rad)

      # Giving a depth for each point
      depth = -10*np.ones(len(d))

      for i in d:
            x1, y1 = point_position(x, y, d, theta)
            xy = [x1, y1, depth]
            xy = np.array(xy)

            xx, yy, dd = xy[0], xy[1], xy[2]
            xy = np.array([xx, yy, dd])

      xy_new = np.zeros((22, 3))

      for i in range(0, len(xx)):
            xy_new[i] = np.array([xx[i], yy[i], dd[i]])

      def distance_between_two_points(x1, y1, x2, y2):
            x = ((x2-x1)**2) - ((y2-y1)**2)
            points_distance = np.sqrt(np.abs(x))
            return points_distance

      for i in range(6, 16):

            # Section -----------------------------------------------------------------------------------------------
            dist_tan = distance_point_from_plane(utmx_tan/1000,utmy_tan/1000, -tan['depth'], normal_ref, xy_new[i])
            resultdist_tan = np.where(dist_tan < 1)
            X_onsection_tan =+ (utmy_tan[resultdist_tan]/1000-xy_new[i][1])*normal_ref[0]-(utmx_tan[resultdist_tan]/1000-xy_new[i][0])*normal_ref[1]

            dist_pre = distance_point_from_plane(utmx_pre/1000,utmy_pre/1000, -catalog_pre['depth'], normal_ref, xy_new[i])
            resultdist_pre = np.where(dist_pre < 1)
            X_onsection_pre =+ (utmy_pre[resultdist_pre]/1000-xy_new[i][1])*normal_ref[0]-(utmx_pre[resultdist_pre]/1000-xy_new[i][0])*normal_ref[1]

            dist_post = distance_point_from_plane(utmx_post/1000,utmy_post/1000, -catalog_post['depth'], normal_ref, xy_new[i])
            resultdist_post = np.where(dist_post < 1)
            X_onsection_post =+ (utmy_post[resultdist_post]/1000-xy_new[i][1])*normal_ref[0]-(utmx_post[resultdist_post]/1000-xy_new[i][0])*normal_ref[1]

            dist_wrev_pre = distance_point_from_plane(utmx_wrev_pre/1000,utmy_wrev_pre/1000, -wrev_pre['source_depth_km'], normal_ref, xy_new[i])
            resultdist_wrev_pre = np.where(dist_wrev_pre < 1)
            X_onsection_wrev_pre =+ (utmy_wrev_pre[resultdist_wrev_pre]/1000-xy_new[i][1])*normal_ref[0]-(utmx_wrev_pre[resultdist_wrev_pre]/1000-xy_new[i][0])*normal_ref[1]

            dist_wrev_post = distance_point_from_plane(utmx_wrev_post/1000,utmy_wrev_post/1000, -wrev_post['source_depth_km'], normal_ref, xy_new[i])
            resultdist_wrev_post = np.where(dist_wrev_post < 1)
            X_onsection_wrev_post =+ (utmy_wrev_post[resultdist_wrev_post]/1000-xy_new[i][1])*normal_ref[0]-(utmx_wrev_post[resultdist_wrev_post]/1000-xy_new[i][0])*normal_ref[1]


            # Plot --------------------------------------------------------------------------------------------------
            plt.figure(figsize=(15,7))
            plt.title(f'Section {i+1}', color='k', fontsize=14, fontweight='bold')
            plt.text(-20, 0.3,'SW', fontsize=12)
            plt.text(19.15, 0.3,'NE', fontsize=12)

            plt.scatter(X_onsection_tan, -tan['depth'].loc[resultdist_tan], marker='.', s=1, c='silver', alpha=0.5, label='Tan et al. catalog')
            plt.scatter(X_onsection_post, -catalog_post['depth'].loc[resultdist_post], marker='.', s=15, c='lime', alpha=1, label='Post-Norcia events', edgecolor='darkgreen', linewidth=0.5)
            plt.scatter(X_onsection_pre, -catalog_pre['depth'].loc[resultdist_pre], marker='.', s=15, c='yellow', alpha=1, label='Pre-Norcia events', edgecolor='darkgoldenrod', linewidth=0.5)
            
            plt.scatter(X_onsection_wrev_pre, -wrev_pre['source_depth_km'].loc[resultdist_wrev_pre], marker='.', s=75, c='fuchsia', alpha=1, edgecolor='purple', linewidth=0.5, label='Wrong events (PRE)')
            plt.scatter(X_onsection_wrev_post, -wrev_post['source_depth_km'].loc[resultdist_wrev_post], marker='.', s=75, c='blue', alpha=1, edgecolor='navy', linewidth=0.5, label='Wrong events (POST)')

            plt.xlabel('Distance [km]', fontsize=12), plt.xlim(-20, 20)
            plt.ylabel('Depth [km]', fontsize=12),plt.ylim(-12, 0)
            
            plt.grid(True, alpha=0.1)

            plt.gca().set_aspect('equal', adjustable='box')
            plt.draw()

            lgnd_len = 5
            lgnd = plt.legend(loc='upper right')
            for l in range(0, lgnd_len):
                  lgnd.legendHandles[l]._sizes = [70]

            # Points ------------------------------------------------------------------------------------------------
            lgnd_len = lgnd_len + 1
            
            if (i+1) == 3:
                  point_stat = distance_between_two_points(xx[i], yy[i], utmx_stat[3]/1000, utmy_stat[3]/1000)
                  plt.scatter(point_stat, 0, c='k', marker='|', s=1000)
                  plt.text(point_stat - 0.7, -1.4, stations['name'][3], fontsize=12)

            elif (i+1) == 4:
                  point_stat = distance_between_two_points(xx[i], yy[i], utmx_stat[3]/1000, utmy_stat[3]/1000)
                  plt.scatter(point_stat, 0, c='k', marker='|', s=1000)
                  plt.text(point_stat - 0.7, -1.4, stations['name'][3], fontsize=12)

            elif (i+1) == 9:
                  point_stat = distance_between_two_points(xx[i], yy[i], utmx_stat[7]/1000, utmy_stat[7]/1000)
                  plt.scatter(-point_stat, 0, c='k', marker='|', s=1000)
                  plt.text(-point_stat - 0.9, -1.4, stations['name'][7], fontsize=12)

            elif (i+1) == 10:
                  point_stat = distance_between_two_points(xx[i], yy[i], utmx_stat[7]/1000, utmy_stat[7]/1000)
                  plt.scatter(-point_stat, 0, c='k', marker='|', s=1000)
                  plt.text(-point_stat - 0.9, -1.4, stations['name'][7], fontsize=12)

                  point_evnt = distance_between_two_points(xx[i], yy[i], utmx_visso/1000, utmy_visso/1000)
                  plt.scatter(point_evnt, -visso_depth, c='red', marker='*', s=300, alpha=0.8, label='Visso mainshock')

                  lgnd = plt.legend(loc='upper right')
                  for l in range(0, lgnd_len):
                        lgnd.legendHandles[l]._sizes = [70]

            elif (i+1) == 11:
                  point_stat = distance_between_two_points(xx[i], yy[i], utmx_stat[2]/1000, utmy_stat[2]/1000)
                  plt.scatter(point_stat, 0, c='k', marker='|', s=1000)
                  plt.text(point_stat - 0.65, -1.4, stations['name'][2], fontsize=12)

                  point_evnt = distance_between_two_points(xx[i], yy[i], utmx_visso/1000, utmy_visso/1000)
                  plt.scatter(point_evnt, -visso_depth, c='red', marker='*', s=300, alpha=0.8)

                  point_evnt = distance_between_two_points(xx[i], yy[i], utmx_visso2/1000, utmy_visso2/1000)
                  plt.scatter(point_evnt, -visso2_depth, c='red', marker='*', s=300, alpha=0.8, label='Visso events')

                  lgnd = plt.legend(loc='upper right')
                  for l in range(0, lgnd_len):
                        lgnd.legendHandles[l]._sizes = [70]

            elif (i+1) == 12:
                  point_stat = distance_between_two_points(xx[i], yy[i], utmx_stat[2]/1000, utmy_stat[2]/1000)
                  plt.scatter(point_stat, 0, c='k', marker='|', s=1000)
                  plt.text(point_stat - 0.65, -1.4, stations['name'][2], fontsize=12)

                  point_evnt = distance_between_two_points(xx[i], yy[i], utmx_visso2/1000, utmy_visso2/1000)
                  plt.scatter(point_evnt, -visso2_depth, c='red', marker='*', s=300, alpha=0.8, label='Visso 2nd event')

                  lgnd = plt.legend(loc='upper right')
                  for l in range(0, lgnd_len):
                        lgnd.legendHandles[l]._sizes = [70]

            elif (i+1) == 14:
                  point_stat = distance_between_two_points(xx[i], yy[i], utmx_stat[0]/1000, utmy_stat[0]/1000)
                  plt.scatter(point_stat, 0, c='k', marker='|', s=1000)
                  plt.text(point_stat - 0.65, -1.4, stations['name'][0], fontsize=12)

                  point_evnt = distance_between_two_points(xx[i], yy[i], utmx_norcia/1000, utmy_norcia/1000)
                  plt.scatter(point_evnt, -norcia_depth, c='red', marker='*', s=500, alpha=1, label='Norcia mainshock')

                  lgnd = plt.legend(loc='upper right')
                  for l in range(0, lgnd_len):
                        lgnd.legendHandles[l]._sizes = [70]

            elif (i+1) == 16:
                  point_evnt = distance_between_two_points(xx[i], yy[i], utmx_amatrice_af/1000, utmy_amatrice_af/1000)
                  plt.scatter(point_evnt, -amatrice_af_depth, c='red', marker='*', s=300, alpha=0.75, label='Amatrice aftershock')

                  lgnd = plt.legend(loc='upper right')
                  for l in range(0, lgnd_len):
                        lgnd.legendHandles[l]._sizes = [70]

            elif (i+1) == 17:
                  point_stat = distance_between_two_points(xx[i], yy[i], utmx_stat[4]/1000, utmy_stat[4]/1000)
                  plt.scatter(-point_stat, 0, c='k', marker='|', s=1000)
                  plt.text(-point_stat - 0.9, -1.4, stations['name'][4], fontsize=12)

            elif (i+1) == 19:
                  point_stat = distance_between_two_points(xx[i], yy[i], utmx_stat[6]/1000, utmy_stat[6]/1000)
                  plt.scatter(point_stat, 0, c='k', marker='|', s=1000)
                  plt.text(point_stat - 0.9, -1.4, stations['name'][6], fontsize=12)

            elif (i+1) == 20:
                  point_stat = distance_between_two_points(xx[i], yy[i], utmx_stat[6]/1000, utmy_stat[6]/1000)
                  plt.scatter(point_stat, 0, c='k', marker='|', s=1000)
                  plt.text(point_stat - 0.9, -1.4, stations['name'][6], fontsize=12)

                  point_stat = distance_between_two_points(xx[i], yy[i], utmx_stat[5]/1000, utmy_stat[5]/1000)
                  plt.scatter(-point_stat, 0, c='k', marker='|', s=1000)
                  plt.text(-point_stat - 0.9, -1.4, stations['name'][5], fontsize=12)

            elif (i+1) == 21:
                  point_stat = distance_between_two_points(xx[i], yy[i], utmx_stat[8]/1000, utmy_stat[8]/1000)
                  plt.scatter(point_stat, 0, c='k', marker='|', s=1000)
                  plt.text(point_stat - 0.9, -1.4, stations['name'][8], fontsize=12)

            elif (i+1) == 22:
                  point_evnt = distance_between_two_points(xx[i], yy[i], utmx_amatrice/1000, utmy_amatrice/1000)
                  plt.scatter(point_evnt, -amatrice_depth, c='red', marker='*', s=300, alpha=0.75, label='Amatrice mainshock')

                  lgnd = plt.legend(loc='upper right')
                  for l in range(0, lgnd_len):
                        lgnd.legendHandles[l]._sizes = [70]

            else:
                  if save == True:
                        plt.savefig(f'./sections_data/images/{station}/section_{i+1}_{station}.jpg', dpi=300, bbox_inches='tight', facecolor=None)
                        continue

            if save == True:            
                  plt.savefig(f'./sections_data/images/{station}/section_{i+1}_{station}.jpg', dpi=300, bbox_inches='tight', facecolor=None)
                  plt.show()

def map_plot(dataframe, station, utm=True, save=False, traces=True):

      tan = pd.read_csv('./sections_data/Amatrice_CAT5.v20210325', delimiter='\s+', header=19)

      tan.columns = ['year', 'month', 'day', 'hour', 'min', 'secmsec', 'lat', 'lon', 'depth', 'EH1', 'EH2', 'AZ', 'EZ', 'Ml', 'Mw', 'ID']
      tan = tan.drop(['EH1', 'EH2', 'AZ', 'EZ'], axis=1)

      df = pd.DataFrame({'year': tan['year'],
                        'month': tan['month'],
                        'day': tan['day'],
                        'hour': tan['hour'],
                        'minute': tan['min'], 
                        'second': tan['secmsec']
      })
      tan['time'] = pd.to_datetime(df, format='%d%m%y %H:%M:%S')
      tan = tan.drop(['year', 'month', 'day', 'hour', 'min', 'secmsec'], axis=1)

      tan = tan.reset_index(drop=True)

      # Importing stations
      stations = {'name': ['NRCA',  'MMO1', 'MC2', 'FDMO', 'T1212', 'T1213', 'T1214', 'T1216', 'T1244'],
                  'lat': [42.83355, 42.899333, 42.911418, 43.0365, 42.751556, 42.724918, 42.759537, 42.890667, 42.75697],
                  'lon': [13.11427, 13.326833, 13.188976, 13.0873, 13.044636, 13.125775, 13.208697, 13.019000, 13.29779],
                  'alt': [927, 957, 2, 550, 869, 860, 1490, 620, 950]}

      stations = pd.DataFrame(data=stations)

      # Importing locations
      locations = {'name': ['Visso', 'Norcia', 'Amatrice', 'Accumoli', 'Campotosto'],
            'lat': [42.9303, 42.7942, 42.628016, 42.694592, 42.553978],
            'lon': [13.0885, 13.0964, 13.292479, 13.245461, 13.370281]}

      locations = pd.DataFrame(data=locations)

      # Importing selected events
      catalog_pre = pd.read_csv('./sections_data/catalog_pre.csv')
      catalog_post = pd.read_csv('./sections_data/catalog_post.csv')

      # Greater events
      grev = pd.read_csv('./sections_data/greater_events.csv')

      # Principal events
      prev = grev
      prev = prev.reset_index(drop=True)
      prev = prev.drop([1, 2, 5, 6, 7, 8]).reset_index(drop=True)

      # Wrong events
      output_catalog = dataframe

      wrev = output_catalog.loc[output_catalog.binary_output != output_catalog.label]
      wrev = wrev.reset_index(drop=True)

      wrev_pre = wrev.loc[wrev.trace_start_time < '2016-10-31 07:41:00']
      wrev_pre = wrev_pre.reset_index(drop=True)

      wrev_post = wrev.loc[wrev.trace_start_time > '2016-10-31 07:41:00']
      wrev_post = wrev_post.reset_index(drop=True)
      
      import pyproj
      utm33 = pyproj.Proj(proj='utm', zone=33, ellps='WGS84', datum='WGS84', units='m')

      # Events
      utmx_tan, utmy_tan = utm33(np.array(tan['lon']), np.array(tan['lat']))
      utmx_pre, utmy_pre = utm33(np.array(catalog_pre['lon']), np.array(catalog_pre['lat']))
      utmx_post, utmy_post = utm33(np.array(catalog_post['lon']), np.array(catalog_post['lat']))

      # Stations and locations
      utmx_stat, utmy_stat = utm33(np.array(stations['lon']), np.array(stations['lat']))
      utmx_loc, utmy_loc = utm33(np.array(locations['lon']), np.array(locations['lat']))

      # Greater events
      utmx_grev, utmy_grev = utm33(np.array(grev['lon']), np.array(grev['lat']))
      utmx_prev, utmy_prev = utm33(np.array(prev['lon']), np.array(prev['lat']))

      # Main events
      amatrice_lon, amatrice_lat, amatrice_depth = 13.2428, 42.7145, 3.276
      utmx_amatrice, utmy_amatrice = utm33(amatrice_lon, amatrice_lat)
      amatrice = np.array([utmx_amatrice/1000, utmy_amatrice/1000, -amatrice_depth])

      amatrice_af_lon, amatrice_af_lat, amatrice_af_depth = 13.15885, 42.80178, 4.113
      utmx_amatrice_af, utmy_amatrice_af = utm33(amatrice_af_lon, amatrice_af_lat)
      amatrice_af = np.array([utmx_amatrice_af/1000, utmy_amatrice_af/1000, -amatrice_af_depth])

      visso_lon, visso_lat, visso_depth = 13.1394, 42.9109, 2.605
      utmx_visso, utmy_visso = utm33(visso_lon, visso_lat)
      visso = np.array([utmx_visso/1000, utmy_visso/1000, -visso_depth])

      visso2_lon, visso2_lat, visso2_depth = 13.12915	, 42.88725, 1.672
      utmx_visso2, utmy_visso2 = utm33(visso2_lon, visso2_lat)
      visso2 = np.array([utmx_visso2/1000, utmy_visso2/1000, -visso2_depth])

      norcia_lon, norcia_lat, norcia_depth = 13.12131, 42.83603, 6.104
      utmx_norcia, utmy_norcia = utm33(norcia_lon, norcia_lat)
      norcia = np.array([utmx_norcia/1000, utmy_norcia/1000, -norcia_depth])

      # Wrong events
      utmx_wrev_pre, utmy_wrev_pre = utm33(np.array(wrev_pre['source_longitude']), np.array(wrev_pre['source_latitude']))
      utmx_wrev_post, utmy_wrev_post = utm33(np.array(wrev_post['source_longitude']), np.array(wrev_post['source_latitude']))

      if utm == True:
            # Prendo come perno norcia e tutti i punti equidistanti 2 km , -18 da una parte e 32 dall'altra
            d = np.arange(-26, 18, 2)   # Numero di punti
            x, y = norcia[0], norcia[1] # Coordinate del perno
            theta, theta_2 = 155, 335   # Strike

            def point_position(x, y, d, theta):
                  # Trova le coordinate dei punti equidistanti 2 km da Norcia
                  theta_rad = np.pi/2 - np.radians(theta)
                  return x + d*np.cos(theta_rad), y + d*np.sin(theta_rad)

            # Per ogni perno di ogni sezione, cioè i punti rossi nella figura sopra, gli do una profodnità di 10
            depth = -10*np.ones(len(d))

            for i in d:
                  x1, y1 = point_position(x, y, d, theta)
                  xy = [x1, y1, depth]
                  xy = np.array(xy)

                  xx, yy, dd = xy[0], xy[1], xy[2]
                  xy = np.array([xx, yy, dd])

            xy_new = np.zeros((22, 3))

            d2 = np.arange(-20, 20, 40)
            ds = np.arange(-25, 47.5, 9)

            th= 60
            th2 = 240
            ths= 155

            for i in xx:
                  xx1, yy1 = point_position(xx,yy,d2,th)
                  xx2, yy2 = point_position(xx,yy,d2,th2)
                  xxs, yys = point_position(norcia[0],norcia[1],ds,ths)

            for i in range(0, len(xx)):
                  xy_new[i] = np.array([xx[i], yy[i], dd[i]])
                  
            fig, ax = plt.subplots(1,1,figsize=(10,10))
            plt.scatter(utmx_tan/1000, utmy_tan/1000, c='silver', marker='.', s=0.5, label='Tan et al. 2021', alpha=0.2)
            plt.scatter(utmx_post/1000, utmy_post/1000, c='lime', edgecolor='darkgreen', linewidth=0.3, marker='.', s=15, label='Post-Norcia events', alpha=0.8)
            plt.scatter(utmx_pre/1000, utmy_pre/1000, c='yellow', edgecolor='darkgoldenrod', linewidth=0.3, marker='.', s=15, label='Pre-Norcia events', alpha=0.8)

            plt.scatter(utmx_wrev_post/1000, utmy_wrev_post/1000, marker='.', s=50, c='blue', alpha=1, edgecolor='navy', linewidth=0.5, label='Wrong events (POST)')
            plt.scatter(utmx_wrev_pre/1000, utmy_wrev_pre/1000, marker='.', s=50, c='fuchsia', alpha=1, edgecolor='purple', linewidth=0.5, label='Wrong events (PRE)')

            plt.scatter(utmx_stat/1000, utmy_stat/1000, c='k', marker='^', s=50, label='Stations')
            plt.scatter(utmx_loc[0:2]/1000, utmy_loc[0:2]/1000, c='k', marker='s', s=10)

            plt.scatter(norcia[0], norcia[1], c='red', marker='*', s=400, label='Principal events')
            plt.scatter(utmx_grev/1000, utmy_grev/1000, c='red', marker='*', s=100, alpha=0.75)

            for n in range(0, len(stations)):
                  ax.text(utmx_stat[n]/1000 + 0.3, utmy_stat[n]/1000 + 0.2, stations['name'][n], color='k')

            for m in range(0, len(locations[0:2])):
                  ax.text(utmx_loc[m]/1000 + 0.3, utmy_loc[m]/1000 + 0.2, locations['name'][m], color='k')
            
            ax.set_xlabel('x [km]', fontsize=12)
            ax.set_ylabel('y [km]', fontsize=12)

            ax.set_aspect('equal')
            ax.grid(True, alpha=0.1)

            ax.set_xlim(323,373)
            ax.set_ylim(4730,4777)

            if traces == True:

                  for i in range(6,16):
                        
                        angle = -155

                        sl = np.tan(np.radians(angle))
                        x = np.array(np.linspace(xx1[i],xx2[i], 24))
                        
                        #dd= np.array([-20,20])
                        #plt.plot(dd[0]+pippo[i],dd[1]+pluto[i])
                        y0 = sl*(x - xx[i]) + yy[i]
                        
                        plt.plot(x, y0, color='k', linewidth=.5)
                        
                        #item = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
                        plt.text(xx2[i]-1, yy2[i]-2.5, str(i+1), fontsize=12, color='w', backgroundcolor='k')

                        if save == True:
                              plt.savefig(f'./sections_data/images/{station}/index_{station}_utm_traces.jpg', dpi=300, bbox_inches='tight', facecolor=None)

            lgnd = plt.legend(loc='upper right')
            for l in range(0, 7):
                  lgnd.legendHandles[l]._sizes = [70]
            
            if save == True:
                  plt.savefig(f'./sections_data/images/{station}/index_{station}_utm_no-traces.jpg', dpi=300, bbox_inches='tight', facecolor=None)

      if utm == False:
            fig, ax = plt.subplots(1,1,figsize=(10,10))
            plt.scatter(tan['lon'], tan['lat'], c='silver', marker='.', s=0.5, label='Tan et al. 2021', alpha=0.2)
            plt.scatter(catalog_post['lon'], catalog_post['lat'], c='lime', edgecolor='darkgreen', linewidth=0.3, marker='.', s=15, label='Post-Norcia events', alpha=0.8)
            plt.scatter(catalog_pre['lon'], catalog_pre['lat'], c='yellow', edgecolor='darkgoldenrod', linewidth=0.3, marker='.', s=15, label='Pre-Norcia events', alpha=0.8)

            plt.scatter(wrev_post['source_longitude'], wrev_post['source_latitude'], marker='.', s=50, c='blue', alpha=1, edgecolor='navy', linewidth=0.5, label='Wrong events (POST)')
            plt.scatter(wrev_pre['source_longitude'], wrev_pre['source_latitude'], marker='.', s=50, c='fuchsia', alpha=1, edgecolor='purple', linewidth=0.5, label='Wrong events (PRE)')

            plt.scatter(stations['lon'], stations['lat'], c='k', marker='^', s=50, label='Stations')
            plt.scatter(locations[0:2]['lon'], locations[0:2]['lat'], c='k', marker='s', s=10)

            plt.scatter(norcia_lon, norcia_lat, c='red', marker='*', s=400, label='Principal events')
            plt.scatter(grev['lon'], grev['lat'], c='red', marker='*', s=100, alpha=0.75)

            for n in range(0, len(stations)):
                  ax.text(stations['lon'][n] + 0.002, stations['lat'][n] + 0.002, stations['name'][n], color='k')

            for m in range(0, len(locations[0:2])):
                  ax.text(locations['lon'][m] + 0.002, locations['lat'][m] + 0.002, locations['name'][m], color='k')

            ax.set_xlabel('Longitude [°]', fontsize=12)
            ax.set_ylabel('Latitude [°]', fontsize=12)
            
            ax.set_aspect('equal')
            plt.grid(True, alpha=0.1)

            plt.xlim(12.9, 13.4)
            plt.ylim(42.70,43.10)      

            lgnd = plt.legend(loc='upper right')
            for l in range(0, 7):
                  lgnd.legendHandles[l]._sizes = [70]
            
            if save == True:
                  plt.savefig(f'./sections_data/images/{station}/index_{station}_lat-lon.jpg', dpi=300, bbox_inches='tight', facecolor=None)
                  
                  
                  
                  


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t,device="cpu"):#, reduce_noise
    """ 
    Takes an earthqk and a timestep as input and 
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)*0.1
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)
    
    
def get_loss_unsupervised(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

@torch.no_grad()
def sample_timestep(x, t, model):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)*0.1
        return model_mean + torch.sqrt(posterior_variance_t) * noise 