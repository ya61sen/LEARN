import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import torch.nn.functional as F

def read_csv_by_chunk(filepath, patient_idx, dtypes=None, chunk_size=1e5):
    filtered_data = []
    if dtypes is not None:
        df_chunks = pd.read_csv(filepath, header=0, index_col=0, chunksize=chunk_size, dtype=dtypes)
    else:
        df_chunks = pd.read_csv(filepath, header=0, index_col=0, chunksize=chunk_size)
    for chunk in df_chunks:
        # Filter the chunk based on patient_idx
        filtered_chunk = chunk[chunk.index.isin(patient_idx)]
        filtered_data.append(filtered_chunk)

    # Concatenate all the filtered chunks into a single DataFrame
    filtered_df = pd.concat(filtered_data)
    return filtered_df

def merge_term_description(df, standard_term_df):
    df['code'] = df['code'].astype(str)
    standard_term_df['code'] = standard_term_df['code'].astype(str)

    merged_df = pd.merge(
        df.reset_index(), 
        standard_term_df[['code_system', 'code', 'code_description']], 
        on=['code_system', 'code'], 
        how='left' 
    )
    return merged_df

# Define a function to format each diagnosis entry
def format_diagnosis(row, date_var_name='date', code_des_var_name='code_description'):
    return f"[{row[date_var_name].strftime('%Y-%m-%d')}] - {row[code_des_var_name]}"

def process_patient(group, date_var_name='date', code_des_var_name='code_description'):
    # iDate: index date
    cutoff_date = group['iDate'].iloc[0]
    filtered_group = group[group[date_var_name] < cutoff_date]
    
    if filtered_group.shape[0] == 0:
        return None
    else:
        # Sort by date and concatenate the diagnoses
        return ' | '.join(filtered_group.sort_values(by=date_var_name).apply(
            lambda row: format_diagnosis(row, date_var_name=date_var_name, code_des_var_name=code_des_var_name), axis=1
            ))

def covert_record_df_by_patient(df, iDate_df, date_var_name='date', code_des_var_name='code_description'):
    iDate_df = iDate_df.reset_index()
    df[date_var_name] = pd.to_datetime(df[date_var_name], format='%Y%m%d')
    iDate_df['iDate'] = pd.to_datetime(iDate_df['iDate'], format='%Y-%m-%d')
    patient_iDate_dict = iDate_df.set_index('patient_id')['iDate'].to_dict()
    df['iDate'] = df['patient_id'].map(patient_iDate_dict)

    # Group by 'patient_id', sort by 'date', and concatenate the diagnosis information
    result_df = df.groupby('patient_id').apply(lambda g: process_patient(g, date_var_name=date_var_name, code_des_var_name=code_des_var_name)).reset_index()

    # Filter out rows where the summary is None
    result_df = result_df[result_df[0].notnull()]

    # Rename columns for clarity
    result_df.columns = ['patient_id', 'summary']
    return result_df.reset_index(drop=True)

def screen_atleast_m_records(df, m=20):
    ehr_data = df['summary']
    len_data = [len(x.split('|')) for x in ehr_data]
    atleast_m = np.array(len_data) >= m
    return df[atleast_m]

def mean_embedding_by_sliding_window(text, model, prefix, window_size=8192, max_length=8192):
    step_size = int(window_size/2)  # Overlap size
    
    text_list = text.split(' ')
    windowed_texts = [text_list[i:i+window_size] for i in range(0, len(text_list), step_size)]
    
    total_embedding_list = []
    for window in windowed_texts:
        embeddings = model.encode([' '.join(window)], instruction=prefix, max_length=max_length)
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        total_embedding_list.append(embeddings_norm.squeeze().detach().cpu().numpy())
    
    mean_embedding = np.array(total_embedding_list).mean(axis=0)
    return mean_embedding





