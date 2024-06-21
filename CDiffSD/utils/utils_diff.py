import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pdb

def create_dataloader(df, feature_columns=['E_channel', 'N_channel', 'Z_channel'], target_columns=['p_arrival_sample', 's_arrival_sample'], trace_name_column=['trace_name'], batch_size=32, shuffle=True, normalize=True, is_noise=False, train_frac=0.7, val_frac=0.15, test_frac=0.15):
    # Step 1: Normalization of columns
    def normalize_multidimensional_row(row):
        '''
        Normalizes the Z, E, and N channel columns for a single row.
        
        Args:
            row (pd.Series): A row of the DataFrame containing the channels to normalize.
            
        Returns:
            pd.Series: A Series containing the normalized channels.
        '''
        Z = np.array(row['Z_channel'])
        E = np.array(row['E_channel'])
        N = np.array(row['N_channel'])

        # Combine the channels into a single array to treat the event as a single entity
        event = np.stack([Z, E, N], axis=0)  # Shape: (3, N)

        # Convert to a PyTorch tensor for normalization
        src = torch.from_numpy(event)

        # Find the maximum absolute value in the entire event to normalize the whole event as a single entity
        src_max = src.abs().max()

        # Normalize the event by its maximum absolute value
        src_norm = src / src_max

        # Convert the normalized tensor back into a numpy array
        event_normalized = src_norm.numpy()

        # Return the channels as separate arrays to keep them distinguishable in the DataFrame
        Z_normalized, E_normalized, N_normalized = event_normalized[0], event_normalized[1], event_normalized[2]

        # Return the normalized channels as a Series
        return pd.Series({'Z_channel': Z_normalized, 'E_channel': E_normalized, 'N_channel': N_normalized})

    # Apply normalization to the DataFrame
    df_normalized = df.apply(normalize_multidimensional_row, axis=1)
    df_normalized['p_arrival_sample'] = df['p_arrival_sample'].values
    df_normalized['s_arrival_sample'] = df['s_arrival_sample'].values
    df_normalized['trace_name'] = df['trace_name'].values

    # Step 2: Splitting data into training, validation, and test sets
    def split_data(df_normalized, train_frac, val_frac, test_frac):
        '''
        Splits the normalized DataFrame into training, validation, and test sets.
        
        Args:
            df_normalized (pd.DataFrame): The normalized DataFrame.
            train_frac (float): Fraction of data for training.
            val_frac (float): Fraction of data for validation.
            test_frac (float): Fraction of data for testing.
            
        Returns:
            tuple: DataFrames for training, validation, and test sets.
        '''
        assert train_frac + val_frac + test_frac == 1.0, "Fractions must sum to 1"
        train_df, temp_df = train_test_split(df_normalized, test_size=1.0 - train_frac, random_state=42)
        val_size = val_frac / (val_frac + test_frac)
        val_df, test_df = train_test_split(temp_df, test_size=1.0 - val_size, random_state=42)
        return train_df, val_df, test_df
    
    # Split the data
    train_df, val_df, test_df = split_data(df_normalized, train_frac, val_frac, test_frac)
    
    # Step 3: Converting columns to arrays
    def convert_column_to_tensor(column):
        '''
        Converts a DataFrame column to a PyTorch tensor.
        
        Args:
            column (pd.Series): The column to convert.
            
        Returns:
            torch.Tensor: The converted tensor.
        '''
        stacked_array = np.stack(column.values)
        return torch.tensor(stacked_array, dtype=torch.float32)
    
    def prepare_tensors(df_normalized):
        '''
        Prepares the feature and target tensors from the DataFrame.
        
        Args:
            df_normalized (pd.DataFrame): The normalized DataFrame.
            
        Returns:
            tuple: TensorDataset of indices, features, and targets, and a dictionary mapping indices to trace names.
        '''
        feature_tensors = [convert_column_to_tensor(df_normalized[col]) for col in feature_columns]
        # Stack along the new dimension (1) to keep channels separate
        features = torch.stack(feature_tensors, dim=1)
        
        length = features.size(0)

        if is_noise:
            p_sample_tensor = torch.zeros(length)
            s_sample_tensor = torch.zeros(length)
        else:
            s_sample_array = np.array(df_normalized['s_arrival_sample'].tolist(), dtype=float).round(1)
            s_sample_array = np.round(s_sample_array, 1)

            p_sample_array = np.array(df_normalized['p_arrival_sample'].tolist(), dtype=float).round(1)
            p_sample_array = np.round(p_sample_array, 1)

            p_sample_tensor = torch.tensor(p_sample_array, dtype=torch.float32)
            s_sample_tensor = torch.tensor(s_sample_array, dtype=torch.float32)

        targets = torch.cat([p_sample_tensor.unsqueeze(1), s_sample_tensor.unsqueeze(1)], dim=1)

        df_normalized = df_normalized.reset_index(drop=True)
        indices = torch.arange(len(df_normalized))
        
        index_to_trace_name = {index: name for index, name in enumerate(df_normalized[trace_name_column])}
        
        return TensorDataset(indices, features, targets), index_to_trace_name
    
    # Step 4: Creating DataLoader instances
    train_dataset, index_to_trace_name = prepare_tensors(train_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    val_dataset, _ = prepare_tensors(val_df)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset, _ = prepare_tensors(test_df)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, index_to_trace_name
