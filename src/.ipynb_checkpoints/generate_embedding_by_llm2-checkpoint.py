import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import gc
from transformers import AutoModel

from utils.generate_embeddings import read_csv_by_chunk, merge_term_description, covert_record_df_by_patient
from utils.generate_embeddings import screen_atleast_m_records, mean_embedding_by_sliding_window

import os
os.environ['HF_HOME']='/storage/group/dxl46/default/private/senyang/huggingface/cache/'

def llm_preprocessing(LIS = 2 * 365, BP = 180, FUP = 5 * 365, data_folder = '../data'):
    # Read data
    sle_ehr_df = pd.read_csv(f'{data_folder}/final_data_yrCut0.txt', sep='\t', header=0, index_col=0)
    
    sle_ehr_df['iDate'] = pd.to_datetime(sle_ehr_df['iDate'], format='%Y-%m-%d')
    sle_ehr_df['date_sle_new'] = pd.to_datetime(sle_ehr_df['date_sle_new'], format='%Y-%m-%d')
    sle_ehr_df['date_rd_new'] = pd.to_datetime(sle_ehr_df['date_rd_new'], format='%Y-%m-%d')
    
    removed_condition_1 = sle_ehr_df['days_till_sle_new'] <= BP
    removed_condition_2 = (sle_ehr_df['age_at_ANAp'] <= 15) | (sle_ehr_df['age_at_ANAp'] > 85)
    removed_condition_3 = sle_ehr_df['diagnosis_dile'] == 1
    removed_condition_4 = sle_ehr_df['diagnosis_cancer'] == 1
    removed_condition_5 = sle_ehr_df['date_rd_new'].notna()
    removed_condition_6 = sle_ehr_df['days_first_event'] > -LIS
    removed_condition_overall = removed_condition_1 | removed_condition_2 | removed_condition_3 |\
                                removed_condition_4 | removed_condition_5 | removed_condition_6 
    sle_ehr_df = sle_ehr_df.loc[~removed_condition_overall, :]
    
    # index date: first ANA+ test
    # Buffer period: Patients who got diagnosed with SLE in this period (index date, index date + BP) were removed
    # Patients diagnosed with SLE before the index date were removed
    case_condition_1 = (sle_ehr_df['days_till_sle_new'] > BP) & (sle_ehr_df['days_till_sle_new'] <= FUP)
    
    # Patients who got diagnosed with SLE in the period > FUP were treated as controls
    # The latest EHR date from the index date should > FUP 
    control_condition_1 = sle_ehr_df['days_till_sle_new'] > FUP
    control_condition_2 = sle_ehr_df['days_last_event'] > FUP
    
    conditions = [case_condition_1, control_condition_1 | control_condition_2]
    sle_ehr_df = sle_ehr_df.copy()
    sle_ehr_df['case_control'] = np.select(conditions, ['Case', 'Control'], default='Removed')
    sle_ehr_df = sle_ehr_df.loc[sle_ehr_df['case_control'] != 'Removed', :]
    
    # Prepare for the text data for LLM
    ## In order to get EHR data with plain text, use the index (patient_id) to subset the raw data for - diagnosis, procedure, medication.
    
    patient_idx = list(sle_ehr_df.index)
    
    standard_term = pd.read_csv(f'{data_folder}/standardized_terminology.csv', header=0, dtype={'unit': str, 'code': str})

    processed_data_save_folder = f'{data_folder}/processed_data'
    os.makedirs(processed_data_save_folder, exist_ok=True)

    # Define file paths
    diagnosis_path = f'{processed_data_save_folder}/diagnosis_by_patient_LIS-{LIS}_BP-{BP}_FUP-{FUP}.csv'
    procedure_path = f'{processed_data_save_folder}/procedure_by_patient_LIS-{LIS}_BP-{BP}_FUP-{FUP}.csv'
    medication_path = f'{processed_data_save_folder}/medication_by_patient_LIS-{LIS}_BP-{BP}_FUP-{FUP}.csv'
    
    # diagnosis
    if os.path.exists(diagnosis_path):
        print("Diagnosis data already processed. Loading from file...")
        diagnosis_by_patient = pd.read_csv(diagnosis_path)
    else:
        print("Processing raw diagnosis data...")
        diagnosis_raw_df = read_csv_by_chunk(f'{data_folder}/diagnosis.csv', patient_idx, dtypes={'code':str})
        diagnosis_raw_df = merge_term_description(diagnosis_raw_df, standard_term)
        diagnosis_by_patient = covert_record_df_by_patient(diagnosis_raw_df, sle_ehr_df)
        
        diagnosis_by_patient.to_csv(diagnosis_path, index=False)

    # procedure
    if os.path.exists(procedure_path):
        print("Procedure data already processed. Loading from file...")
        procedure_by_patient = pd.read_csv(procedure_path)
    else:
        print("Processing raw procedure data...")
        procedure_raw_df = read_csv_by_chunk(f'{data_folder}/procedure.csv', patient_idx, dtypes={'code':str})
        procedure_raw_df = merge_term_description(procedure_raw_df, standard_term)
        procedure_by_patient = covert_record_df_by_patient(procedure_raw_df, sle_ehr_df)
        
        procedure_by_patient.to_csv(procedure_path, index=False)

    # medication
    if os.path.exists(medication_path):
        print("Medication data already processed. Loading from file...")
        medication_by_patient = pd.read_csv(medication_path)
    else:
        print("Processing raw medication data...")
        medication_raw_df = read_csv_by_chunk(f'{data_folder}/medication_drug.csv', patient_idx, dtypes={'code':str})
        
        ## process medication data separately
        medication_ndc_raw_df = medication_raw_df.loc[medication_raw_df['code_system'] == 'NDC']
        medication_rxnorm_raw_df = medication_raw_df.loc[medication_raw_df['code_system'] == 'RxNorm']
        
        # NDC
        # ---------------------- Step 1: Load RXNSAT.RRF (NDC to RxNorm Mapping) ----------------------
        rxnsat_columns = [
            'RXCUI', 'LUI', 'SUI', 'RXAUI', 'STYPE', 'CODE', 
            'ATUI', 'SATUI', 'ATN', 'SAB', 'ATV', 'SUPPRESS', 'CVF'
        ]
        
        # Load RXNSAT.RRF, which contains mappings between different drug codes and RxNorm identifiers
        rxnsat = pd.read_csv(f'{data_folder}/RxNorm_full_12022024/rrf/RXNSAT.RRF', delimiter='|', 
                             header=None, index_col=False, names=rxnsat_columns, low_memory=False)
        
        ndc_to_rxcui = rxnsat[(rxnsat['ATN'] == 'NDC')]
        ndc_to_rxcui = ndc_to_rxcui[~ndc_to_rxcui['ATV'].str.contains('-')]
        ndc_list = medication_ndc_raw_df['code'].values
        ndc_to_rxcui_filtered = ndc_to_rxcui[ndc_to_rxcui['ATV'].isin(ndc_list)]
        
        # ---------------------- Step 2: Load RXNCONSO.RRF (RxNorm Concept Descriptions) ----------------------
        
        rxnconso_columns = [
            'RXCUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'RXAUI', 
            'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE', 'STR', 'SRL', 
            'SUPPRESS', 'CVF'
        ]
        
        # Load RXNCONSO.RRF, which contains concept descriptions for RxNorm identifiers
        rxnconso = pd.read_csv(f'{data_folder}/RxNorm_full_12022024/rrf/RXNCONSO.RRF', delimiter='|', 
                               header=None, index_col=False, names=rxnconso_columns, low_memory=False)
        
        ndc_mapped_drugs = ndc_to_rxcui_filtered.merge(rxnconso, on='RXCUI', how='left')
        
        # ---------------------- Step 3: Extract RxNorm Descriptions for RxNorm-coded Medications ----------------------
        
        medication_rxnorm_raw_df.loc[:, 'code'] = medication_rxnorm_raw_df['code'].astype(int)
        rxnorm_list = medication_rxnorm_raw_df['code'].values
        rxnorm_filtered = rxnconso[rxnconso['RXCUI'].isin(rxnorm_list)]
        
        medication_ndc_merged_df = pd.merge(medication_ndc_raw_df.reset_index(), ndc_mapped_drugs.drop_duplicates(subset=['ATV'], keep='first').loc[:, ['RXCUI', 'ATV', 'STR']], 
                                            left_on='code', right_on='ATV', how='left')
        medication_ndc_merged_df = medication_ndc_merged_df.drop(columns=['ATV'])
        
        # RxNorm
        medication_rxnorm_merged_df = pd.merge(medication_rxnorm_raw_df.reset_index(), rxnorm_filtered.drop_duplicates(subset=['RXCUI'], keep='first').loc[:, ['RXCUI', 'STR']], 
                                            left_on='code', right_on='RXCUI', how='left')
        
        medication_raw_df = pd.concat([medication_ndc_merged_df, medication_rxnorm_merged_df], axis=0)
        medication_raw_df['RXCUI'] = medication_raw_df['RXCUI'].astype('Int64')
        
        medication_by_patient = covert_record_df_by_patient(medication_raw_df, sle_ehr_df, date_var_name='start_date', code_des_var_name='STR')
        
        medication_by_patient.to_csv(medication_path, index=False)

    return sle_ehr_df, diagnosis_by_patient, procedure_by_patient, medication_by_patient

def generate_embeddings(data_by_patient, aggregated_embedding_path, final_embedding_path, model, prefix):
    """
    Generates and updates embeddings while maintaining an aggregated file for future cohorts.
    """
    
    # Load existing aggregated embeddings if available
    if os.path.exists(aggregated_embedding_path):
        print(f"Loading existing aggregated embeddings from {aggregated_embedding_path}")
        aggregated_df = pd.read_csv(aggregated_embedding_path, index_col=0)
    else:
        aggregated_df = pd.DataFrame()

    # Convert patient IDs to strings (ensure consistency)
    patient_ids = data_by_patient['patient_id'].astype(str)
    
    # Identify missing patient IDs (those not in aggregated_df)
    missing_patients = patient_ids[~patient_ids.isin(aggregated_df.index)]
    
    # Generate embeddings only for missing patients
    if not missing_patients.empty:
        new_embeddings = []
        
        for i, patient_id in enumerate(missing_patients):
            if i % 500 == 0:
                print(f"Processing - Iteration {i}")

            text = data_by_patient.loc[data_by_patient['patient_id'] == patient_id, 'summary'].values[0]
            embedding = mean_embedding_by_sliding_window(text, model, prefix)
            new_embeddings.append(embedding)

        new_embedding_df = pd.DataFrame(np.vstack(new_embeddings), index=missing_patients)
        new_embedding_df.columns = new_embedding_df.columns.astype(str)
        updated_aggregated_df = pd.concat([aggregated_df, new_embedding_df], axis=0)

        # Save updated aggregated embeddings (overwrite old file)
        updated_aggregated_df.to_csv(aggregated_embedding_path)
        print(f"Aggregated embeddings updated at {aggregated_embedding_path}. Total records: {len(updated_aggregated_df)}")
    else:
        print(f"No new embeddings generated. Using existing aggregated embeddings.")
        updated_aggregated_df = aggregated_df

    final_embedding_df = updated_aggregated_df.loc[patient_ids]
    final_embedding_df.to_csv(final_embedding_path)
    print(f"Final embeddings for the cohort saved at {final_embedding_path}. Total patients: {len(final_embedding_df)}")
    return final_embedding_df
    
def generate_embedding_by_NV_Embed_v2(diagnosis_by_patient, procedure_by_patient, medication_by_patient,
                                      LIS = 2 * 365, BP = 180, FUP = 5 * 365,
                                      data_folder='../data'):
    diagnosis_by_patient = screen_atleast_m_records(diagnosis_by_patient)
    procedure_by_patient = screen_atleast_m_records(procedure_by_patient)
    medication_by_patient = screen_atleast_m_records(medication_by_patient, m=0)
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    local_model_path = "/storage/group/dxl46/default/private/senyang/huggingface/cache/hub/models--nvidia--NV-Embed-v2/snapshots/5130cf1daf847c1bacee854a6ef1ca939e747fb2"
    # load model with tokenizer
    model = AutoModel.from_pretrained(local_model_path, trust_remote_code=True,
                                    #   device_map = 'auto'
                                      torch_dtype=torch.float16
                                      ).to(DEVICE)
    
    # Each query needs to be accompanied by an corresponding instruction describing the task.
    task_name_to_instruct = {
        "diagnosis": "Analyze this patient's diagnosis history up to the date prior to their ANA-positive test. Each entry includes the date and diagnosis in chronological order. Encode the data to identify key patterns and trends in the patient's health conditions that could contribute to predicting the development of systemic lupus erythematosus (SLE).",
        "procedure": "Analyze this patient's procedure history up to the date prior to their ANA-positive test. Each entry includes the date and name of the procedure in chronological order. Encode the data to highlight key patterns and trends in the patient's medical interventions that could contribute to predicting the development of systemic lupus erythematosus (SLE).",
        "medication": "Analyze this patient's medication history up to the date prior to their ANA-positive test. Each entry includes the date and the medication in chronological order. Extract and encode only the drug names (ignoring dosage, form, or other details) to identify patterns in medication use that may contribute to predicting the development of systemic lupus erythematosus (SLE)."
    }
    d_prefix = "Instruct: "+task_name_to_instruct["diagnosis"]+"\nDiagnosis history: "
    p_prefix = "Instruct: "+task_name_to_instruct["procedure"]+"\nProcedure history: "
    m_prefix = "Instruct: "+task_name_to_instruct["medication"]+"\nMedication history: "

    embedding_save_folder = f'{data_folder}/embeddings'
    os.makedirs(embedding_save_folder, exist_ok=True)

    # Define file paths
    diagnosis_embedding_path = f'{embedding_save_folder}/diagnosis_embedding_by_NV-Embed-v2_LIS-{LIS}_BP-{BP}_FUP-{FUP}.csv'
    procedure_embedding_path = f'{embedding_save_folder}/procedure_embedding_by_NV-Embed-v2_LIS-{LIS}_BP-{BP}_FUP-{FUP}.csv'
    medication_embedding_path = f'{embedding_save_folder}/medication_embedding_by_NV-Embed-v2_LIS-{LIS}_BP-{BP}_FUP-{FUP}.csv'

    # aggregated embedding files
    agg_diagnosis_emb_path = f'{embedding_save_folder}/agg_diagnosis_embedding_by_NV-Embed-v2.csv'
    agg_procedure_emb_path = f'{embedding_save_folder}/agg_procedure_embedding_by_NV-Embed-v2.csv'
    agg_medication_emb_path = f'{embedding_save_folder}/agg_medication_embedding_by_NV-Embed-v2.csv'

    diagnosis_df = generate_embeddings(diagnosis_by_patient, agg_diagnosis_emb_path, diagnosis_embedding_path, model, d_prefix)
    procedure_df = generate_embeddings(procedure_by_patient, agg_procedure_emb_path, procedure_embedding_path, model, p_prefix)
    medication_df = generate_embeddings(medication_by_patient, agg_medication_emb_path, medication_embedding_path, model, m_prefix)

    return diagnosis_df, procedure_df, medication_df







