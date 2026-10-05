"""
Shared utilities for processing TS data
"""

import numpy as np
import torch

def standardize_piece(df_piece):
    """
    Standardizes a chronological piece of the dataframe
      by subtracting the mean and dividing by std

    Args:
        df_piece (pandas dataframe)
    """

    # column_indices = {name: i for i, name in enumerate(df_piece.columns)}
    mean = df_piece.mean()
    std = df_piece.std()
    return (df_piece - mean) / std

def create_sequences_single_step(data, seq_length: int):
    """
    Creates sequences for single step prediction

    Args:
        data (array-like): The input data.
        seq_length (int): The length of each training sequence.

    Returns:
        np.ndarray: Array of input sequences.
        np.ndarray: Array of target sequences (half the length of input sequences).
    """
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])                        # Input sequence
        targets.append(data[i+seq_length]) # Target is a single step

    return np.array(sequences), np.array(targets)

def create_sequences_mutlistep(data, seq_length: int):
    """
    Creates sequences and their corresponding target sequences from the input data.
    The target sequence is 1/3 the size of the training sequence.

    Args:
        data (array-like): The input data.
        seq_length (int): The length of each training sequence.

    Returns:
        np.ndarray: Array of input sequences.
        np.ndarray: Array of target sequences (half the length of input sequences).
    """
    target_length = seq_length // 3  # Target is 1/3 the size of the training sequence

    if len(data) < seq_length + target_length:
        raise ValueError("Data length must be at least seq_length + target_length.")

    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length - target_length + 1):
        sequences.append(data[i:i+seq_length])                        # Input sequence
        targets.append(data[i+seq_length:i+seq_length+target_length]) # Target sequence

    return np.array(sequences), np.array(targets)

def make_sequence_tensor(sequence, device):
    """
    Make a pytorch Tensor from a numpy sequence

    Args:
        sequence (np.ndarray): Input sequence
        device : pytorch device
    
    Returns:
        torch.FloatTensor with dim (batch, seq_length, features)
    """
    return torch.FloatTensor(sequence).to(device)
