import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def ml_prediction_pipeline(predict_method, predictor_random_seed=123):
    if predict_method == 'logistic':
        from sklearn.linear_model import LogisticRegression
        predict_pipeline = make_pipeline(StandardScaler(), 
                                        LogisticRegression(random_state=predictor_random_seed,
                                                            penalty='elasticnet', solver='saga', l1_ratio=0.5))
    elif predict_method == 'svm':
        from sklearn.svm import SVC
        predict_pipeline = make_pipeline(StandardScaler(), 
                                        SVC(gamma='auto', random_state=predictor_random_seed, probability=True))
    elif predict_method == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        predict_pipeline = make_pipeline(StandardScaler(), 
                                        RandomForestClassifier(random_state=predictor_random_seed))
    elif predict_method == 'xgboost':
        from xgboost import XGBClassifier
        predict_pipeline = make_pipeline(StandardScaler(), 
                                        XGBClassifier(eval_metric='auc', random_state=predictor_random_seed))
    else:
        raise ValueError("Please choose a prediction method from: {'logistic', 'svm', 'rf', 'xgboost'}!")
    
    return predict_pipeline

def fit_predict_from_ml_pipeline(X_train, y_train, X_test, y_test, classes, 
                                 predict_method, save_dict, predictor_random_seed=123,
                                 model_save_path=None):
    """
    Builds, fits, makes predictions and saves the model pipeline if a save path is provided.
    
    Parameters:
      - X_train, y_train: Training data.
      - X_test, y_test: Testing data.
      - classes: Array or list used to map labels.
      - predict_method: One of {'logistic', 'svm', 'rf', 'xgboost'}.
      - save_dict: Dictionary in which the metrics and predictions are saved.
      - predictor_random_seed: Seed for reproducibility.
      - model_save_path: (Optional) File path (including filename, e.g., 'my_model.pkl')
                          where the fitted model will be saved.
    
    Returns:
      - predict_pipeline: The fitted prediction pipeline.
      - y_pred_proba, y_pred, y_test_true, accuracy, conf_mat, auc: Corresponding prediction outputs and evaluation metrics.
    """
    predict_pipeline = ml_prediction_pipeline(predict_method=predict_method,
                                              predictor_random_seed=predictor_random_seed)
    predict_pipeline.fit(X_train, y_train)

    # Save the fitted model if a save path is provided
    if model_save_path is not None:
        import joblib
        joblib.dump(predict_pipeline, model_save_path)
        print(f"Model saved to {model_save_path}")
    
    y_test_true = classes[y_test]
    accuracy = predict_pipeline.score(X_test, y_test)
    y_pred = predict_pipeline.predict(X_test)
    pred_prob = predict_pipeline.predict_proba(X_test)
    from sklearn.metrics import confusion_matrix
    conf_mat = pd.DataFrame(confusion_matrix(y_test_true, classes[y_pred]),
                            index=['true: {}'.format(x) for x in classes],
                            columns=['pred: {}'.format(x) for x in classes])
    
    from sklearn.metrics import roc_auc_score
    y_pred_proba = pred_prob[:,1] if pred_prob.shape[1]==2 else pred_prob
    auc = roc_auc_score(y_test, y_pred_proba)

    save_dict[predict_method].append({
        'patient_ids': X_test.index.tolist(),
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
        'y_test_true': y_test_true,
        'accuracy': accuracy,
        'confusion_matrix': conf_mat,
        'auc': auc
    })

    return predict_pipeline, y_pred_proba, y_pred, y_test_true, accuracy, conf_mat, auc

def load_and_predict_from_saved_pipeline(model_save_path, X_test, y_test, classes,
                                         predict_method, save_dict):

    import joblib
    from sklearn.metrics import confusion_matrix, roc_auc_score
    
    # Load the saved model pipeline from the specified file
    loaded_pipeline = joblib.load(model_save_path)
    print(f"Model loaded from {model_save_path}")
    
    y_test_true = classes[y_test]
    accuracy = loaded_pipeline.score(X_test, y_test)
    y_pred = loaded_pipeline.predict(X_test)
    pred_prob = loaded_pipeline.predict_proba(X_test)
    
    conf_mat = pd.DataFrame(confusion_matrix(y_test_true, classes[y_pred]),
                            index=['true: {}'.format(x) for x in classes],
                            columns=['pred: {}'.format(x) for x in classes])
    
    y_pred_proba = pred_prob[:, 1] if pred_prob.shape[1] == 2 else pred_prob
    auc = roc_auc_score(y_test, y_pred_proba)
    
    save_dict[predict_method].append({
        'patient_ids': X_test.index.tolist(),
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
        'y_test_true': y_test_true,
        'accuracy': accuracy,
        'confusion_matrix': conf_mat,
        'auc': auc
    })
    
    return loaded_pipeline, y_pred_proba, y_pred, y_test_true, accuracy, conf_mat, auc