import warnings
warnings.simplefilter("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import argparse
import torch.nn as nn
import random
import pickle

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

from models.ml_models import ml_prediction_pipeline, fit_predict_from_ml_pipeline
from models.mlpnn import SLENet, SLEDataset, train, predict_mlp
from utils.plotting import plot_loss, plot_roc, plot_cm
from generate_embedding_by_llm2 import llm_preprocessing, generate_embedding_by_NV_Embed_v2
from utils.utils_funcs import AverageMeter

def setup_seed(seed=123):
    """Ensure reproducibility by setting random seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_embedding_data(LIS, BP, FUP):
    """Preprocess and generate embeddings."""
    sle_ehr_df, diagnosis_by_patient, procedure_by_patient, medication_by_patient = llm_preprocessing(LIS=LIS, BP=BP, FUP=FUP)
    print('--- LLM preprocessing finished! ---')

    diagnosis_df, procedure_df, medication_df = generate_embedding_by_NV_Embed_v2(
        diagnosis_by_patient, procedure_by_patient, medication_by_patient, LIS=LIS, BP=BP, FUP=FUP
    )
    print('--- LLM embedding generation finished! ---')

    return sle_ehr_df, diagnosis_df, procedure_df, medication_df

def preprocess_data(sle_ehr_df, diagnosis_df, procedure_df, medication_df, include_med_emb=True):
    """Clean and preprocess dataset for training."""
    sle_ehr_df_clean = sle_ehr_df.drop(columns=['date_ana', 'n_ana', 'iDate', 'date_rd', 'date_sle', 
                                             'max_date_sle', 'max_date_sle_new', 'min_value_date', 
                                             'max_value_date', 'yob', 'age_at_ANAp', 'date_rd_new',
                                             'date_sle_new', 'days_till_sle_new', 'max_date_sle_new',
                                             'days_till_sle_new_max', 'n_sle', 'n_sle_new', 'diagnosis_dile',
                                             'diagnosis_cancer', 'days_first_event', 'days_last_event',
                                             'age_at_sle_new', 'n_rd', 'n_rd_new', 'days_till_sle',
                                             'days_till_sle_max', 'age_at_sle'])
    sle_ehr_df_clean['cate_by_sex'] = sle_ehr_df_clean[['sex', 'case_control']].apply(lambda x: f'{x.iloc[0]}_{x.iloc[1]}', axis=1)
    X = pd.get_dummies(sle_ehr_df_clean.drop(columns=['case_control', 'cate_by_sex']), drop_first=True)
    X = X.fillna(0)
    y = sle_ehr_df_clean['case_control']

    if include_med_emb:
        diagnosis_df = diagnosis_df.dropna(how='all').dropna(axis=1, how='all')
        procedure_df = procedure_df.dropna(how='all').dropna(axis=1, how='all')
        medication_df = medication_df.dropna(how='all').dropna(axis=1, how='all')
        
        indexes = list(set(diagnosis_df.index) & set(procedure_df.index) & set(medication_df.index) & set(X.index))
        indexes.sort()
        
        X = pd.concat([X.loc[indexes, :], 
                       diagnosis_df.loc[indexes, :], 
                       procedure_df.loc[indexes, :], 
                       medication_df.loc[indexes, :]], axis=1)
        y = y.loc[indexes]
    else:
        diagnosis_df = diagnosis_df.dropna(how='all').dropna(axis=1, how='all')
        procedure_df = procedure_df.dropna(how='all').dropna(axis=1, how='all')
    
        indexes = list(set(diagnosis_df.index) & set(procedure_df.index) & set(X.index))
        indexes.sort()
        X = pd.concat([X.loc[indexes, :], diagnosis_df.loc[indexes, :], procedure_df.loc[indexes, :]], axis=1)
        y = y.loc[indexes]
    return X, y, include_med_emb

def train_model(X, y, LIS, BP, FUP, n_splits=5, N_EPOCH=60, BATCH_SIZE=64, include_med_emb=True,
                result_save_path='./results', random_state=123, auc_summary=True):
    setup_seed(random_state)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    scaler = StandardScaler()
    X_array = scaler.fit_transform(X)
    X_std = pd.DataFrame(X_array, columns=X.columns, index=X.index)
    
    # Encode labels
    le = LabelEncoder()
    le.fit(y)
    y_encoded = le.transform(y)
    classes = le.classes_
    label_dim = len(classes)

    # Cross-validation setup
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    hist_all_folds = []
    predictions_all_folds = {'svm': [],
                             'rf': [],
                             'xgboost': [],
                             'mlp': []}
    include_med_emb_text = 'with_med_emb' if include_med_emb else 'no_med_emb'
    model_save_path = f'{result_save_path}/LIS-{LIS}_BP-{BP}_FUP-{FUP}/count_data_n_embeddings_by_NV-Embed-v2_{include_med_emb_text}/'
    os.makedirs(model_save_path, exist_ok=True)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_std, y_encoded)):
        print(f'Fold {fold + 1}/{kf.n_splits}')
    
        # Split the data for this fold
        X_train, X_test = X_std.iloc[train_idx], X_std.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    
        class_counts = np.unique(y_train, return_counts=True)[1]
        loss_pos_weight = class_counts[0]/class_counts[1]
        print(f'pos_weight:', loss_pos_weight)
    
        from imblearn.over_sampling import SMOTE, SVMSMOTE
        re_sampler = SMOTE(random_state=random_state)
        X_train, y_train = re_sampler.fit_resample(X_train, y_train)
        print('Over sampling finished!')
    
        ## SVM
        print('SVM:')
        predict_pipeline_svm, y_pred_proba_svm, y_pred_svm, y_test_true_svm, accuracy_svm, conf_mat_svm, auc_svm =\
            fit_predict_from_ml_pipeline(X_train, y_train, X_test, y_test, classes,
                                        predict_method='svm', save_dict=predictions_all_folds,
                                        model_save_path=os.path.join(model_save_path, 
                                                                     f'LIS-{LIS}_BP-{BP}_FUP-{FUP}_SVM_fold_{fold + 1}.pkl'))
        print('SVM finished.')
    
        ## RF
        print('RF:')
        predict_pipeline_rf, y_pred_proba_rf, y_pred_rf, y_test_true_rf, accuracy_rf, conf_mat_rf, auc_rf =\
            fit_predict_from_ml_pipeline(X_train, y_train, X_test, y_test, classes,
                                        predict_method='rf', save_dict=predictions_all_folds,
                                        model_save_path=os.path.join(model_save_path, 
                                                                     f'LIS-{LIS}_BP-{BP}_FUP-{FUP}_RF_fold_{fold + 1}.pkl'))
        print('RF finished.')
    
        ## XGBOOST
        print('XGBOOST:')
        predict_pipeline_xgb, y_pred_proba_xgb, y_pred_xgb, y_test_true_xgb, accuracy_xgb, conf_mat_xgb, auc_xgb =\
            fit_predict_from_ml_pipeline(X_train, y_train, X_test, y_test, classes,
                                        predict_method='xgboost', save_dict=predictions_all_folds,
                                        model_save_path=os.path.join(model_save_path, 
                                                                     f'LIS-{LIS}_BP-{BP}_FUP-{FUP}_XGBoost_fold_{fold + 1}.pkl'))
        print('XGBOOST finished.')
    
        ## MLP
        print('MLP:')
    
        train_data = SLEDataset(np.array(X_train).astype('float'), np.array(y_train))
        test_data = SLEDataset(np.array(X_test).astype('float'), np.array(y_test))
    
        if X_train.shape[0]%BATCH_SIZE == 1:
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
        else:
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        if include_med_emb:
            classifier = SLENet(input_dim=X_train.shape[1], 
                                net_hidden_structure = [4096, 1024, 256, 64, 16, 4], 
                                dropout_rates=[0.8, 0.6, 0.6, 0.6, 0.6, 0.6])
        else:
            classifier = SLENet(input_dim=X_train.shape[1], 
                            net_hidden_structure = [2048, 512, 128, 16, 4], 
                            dropout_rates=[0.8, 0.6, 0.6, 0.6, 0.6])
        classifier.to(device)
        # print('classifier:', classifier)
        # criterion = FocalLoss(gamma=2.0, alpha=.99) 
        # criterion = DiceLoss()
    
        # pos_weight: class_counts[classes[0]]/class_counts[classes[1]]
        criterion = nn.BCEWithLogitsLoss() # pos_weight=torch.tensor(loss_pos_weight)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay = 0.0001)
        # optimizer = torch.optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9, weight_decay = 0.01)
    
        # =========================
        hist = {'train_loss': [], 'test_loss': []}
    
        for epoch in range(N_EPOCH):
            # Train
            train_loss = train(epoch, classifier, criterion, optimizer, train_loader, device=device, disable=True)
            hist['train_loss'].append(train_loss)
    
            # test
            classifier.eval()
            test_losses = AverageMeter()
    
            with torch.no_grad():
                for idx, (data, target) in enumerate(test_loader):
                    batch_size = data.size(0)
                    data, target = data.to(device), target.to(device)
    
                    # ===================forward=====================
                    outputs = classifier(data)
                    test_loss = criterion(outputs.squeeze(-1), target.float())
                    # ===================meters======================
                    test_losses.update(test_loss.item(), batch_size)
    
                hist['test_loss'].append(test_losses.avg)
                # print('Epoch {} \ttrain_loss\t{}\ttest_loss\t{}'.format(epoch, train_loss, test_losses.avg))
    
        # Save history for this fold
        hist_all_folds.append(hist)
    
        # from utils import plot_loss
        plot_loss(hist, keys=list(hist.keys()), figsize=(12,6))
    
        # Predict and evaluate for this fold
        y_pred_proba, y_pred, y_test_true, accuracy, confusion_mat, auc = \
            predict_mlp(classifier, X_test, y_test, classes, device=device)
        
        # Store prediction results for each fold
        predictions_all_folds['mlp'].append({
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred,
            'y_test_true': y_test_true,
            'accuracy': accuracy,
            'confusion_matrix': confusion_mat,
            'auc': auc
        })
    
        os.makedirs(model_save_path, exist_ok=True)
    
        # LIS: Length of time in the system (in days)
        # BP: Buffer period (in days)
        # FUP: Follow up period (in days)
        torch.save({
                    'fold': fold,
                    'epoch': N_EPOCH,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': hist,
                    'predictions': predictions_all_folds['mlp'][-1]
                    }, os.path.join(model_save_path, f'LIS-{LIS}_BP-{BP}_FUP-{FUP}_Counts_NN_fold_{fold + 1}.pth'))
        print('MLP finished.')
    
    # Save history and predictions for all folds in a separate file
    with open(os.path.join(model_save_path, f'predictions_all_folds_LIS-{LIS}_BP-{BP}_FUP-{FUP}.pkl'), 'wb') as f:
        pickle.dump({
            'hist_all_folds': hist_all_folds,
            'predictions_all_folds': predictions_all_folds
        }, f)
    
    print("Saved combined predictions and history for all folds in .pkl format.")
    print("Training complete. Results saved.")

    if auc_summary:

        auc_results = []

        print("\n--- AUC Scores Summary ---")
        for pred_method, pred_list in predictions_all_folds.items():
            print(f'Prediction Method: {pred_method}')
            auc_scores = [pred['auc'] for pred in pred_list if 'auc' in pred]
    
            if auc_scores:
                mean_auc = np.mean(auc_scores)
                std_auc = np.std(auc_scores)
                print(f'  - Average AUC: {mean_auc:.4f}')
                print(f'  - Standard Deviation: {std_auc:.4f}')
    
                # Store results in a list (to later convert into a DataFrame)
                auc_results.append({
                    "Prediction Method": pred_method,
                    "Average AUC": mean_auc,
                    "Standard Deviation": std_auc
                })
            else:
                print(f'  - No AUC scores found for {pred_method}')
        
        auc_df = pd.DataFrame(auc_results)
        auc_df.to_csv(f"{model_save_path}/auc_summary.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--LIS", type=int, default=2 * 365)
    parser.add_argument("--BP", type=int, default=180)
    parser.add_argument("--FUP", type=int, default=5 * 365)
    parser.add_argument("--include_med_emb", action="store_true", help="Include medication embeddings in the model training.")
    args = parser.parse_args()

    # Print selected parameters
    print(f"\n--- Training Configuration ---")
    print(f"LIS (Length in System): {args.LIS} days")
    print(f"BP (Buffer Period): {args.BP} days")
    print(f"FUP (Follow-Up Period): {args.FUP} days\n")
    print(f"Include Medication Embeddings: {args.include_med_emb}\n")
    
    sle_ehr_df, diagnosis_df, procedure_df, medication_df = generate_embedding_data(args.LIS, args.BP, args.FUP)
    X, y, include_med_emb = preprocess_data(sle_ehr_df, diagnosis_df, procedure_df, medication_df, args.include_med_emb)
    train_model(X, y, args.LIS, args.BP, args.FUP, include_med_emb=include_med_emb)











