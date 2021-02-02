import pandas as pd 
import numpy as np 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from itertools import combinations 
from sklearn.metrics import accuracy_score

def reliability_model(fold, x_train, x_test, y_train, y_test, model, key, diz_results):
    """
    Compute reliability for each model. F1 score is the weight of the reliability
    """
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    metrics = precision_recall_fscore_support(y_test,prediction, average=None)
    
    diz_results[key]['Fscore0'].append(metrics[2][0])
    diz_results[key]['Fscore1'].append(metrics[2][1])
    
    if fold == 9:
        weights_reliability = pd.DataFrame.from_dict(diz_results, orient='index')
        weights_reliability['F0_weights'] = weights_reliability['Fscore0'].apply(lambda x: np.mean(x))
        weights_reliability['F1_weights'] = weights_reliability['Fscore1'].apply(lambda x: np.mean(x))
        weights_reliability.drop(['Fscore0', 'Fscore1'], axis = 1, inplace = True)
        return weights_reliability
        
def fitting_final(model, key, x, y, x_test, y_test, weights):
    '''
    Compute final probability distribution on a validation set, weighted by reliability of each model
    '''
    model.fit(x,y)
    prediction_proba = model.predict_proba(x_test)
    prediction_proba0 = np.expand_dims(prediction_proba[:,0] * weights.loc[key,:]['F0_weights'],1)
    prediction_proba1 = np.expand_dims(prediction_proba[:,1] * weights.loc[key,:]['F1_weights'],1)
    prediction_weighted = np.concatenate([prediction_proba0, prediction_proba1], axis = 1)
    return prediction_weighted

def create_combinations(list_name_model):
    """
    Create all possible combination 2^n - n (excluding empty combination)
    """
    list_of_combinations = []
    for i in range(1,len(list_name_model)+1):
        for j in combinations(list_name_model, i):
            list_of_combinations.append(list(j))
    return list_of_combinations

def BMA(model_diz, X_train_pp_pos_pol, y):
    """
    Cross validation for finding the right weights for each model, testing all possible combination with an external validation set
    return a dictionary with weights for each model, results in terms of accuracy of the ensemble method, metrics score related on each combination
    """
    strat = StratifiedKFold(n_splits = 10)
    diz_accuracy = {}
    diz_metrics = {}
    diz_predictions = {}
    outputs = {}
    model_diz_results = {i:{'Fscore0':[], 'Fscore1':[]} for i in model_diz.keys()}
    X_train, x_val,Y_train, y_val = train_test_split(X_train_pp_pos_pol, y, test_size = 0.1, stratify = y)
    
    for fold, (train_index, test_index) in enumerate(strat.split(X_train, Y_train)):
        print("Fold number {}".format(fold + 1), end = '\r')
        x_train = X_train[train_index]
        y_train = Y_train[train_index]
        x_test = X_train[test_index]
        y_test = Y_train[test_index]
        for key, model in model_diz.items():
            if fold != 9:
                reliability_model(fold, x_train, x_test, y_train, y_test, model, key, model_diz_results)
            else:
                weights_reliability = reliability_model(fold, x_train, x_test, y_train, y_test, model, key, model_diz_results)

    for key, model in model_diz.items():
        diz_predictions[key] = fitting_final(model, key, X_train, Y_train, x_val, y_val, weights_reliability)
        
    combo = create_combinations(list(model_diz.keys()))
    for elements in combo:
        appoggio = []
        for keys in elements:
            appoggio.append(np.array(diz_predictions[keys]))
        probability_sum = np.sum(np.array(appoggio), axis = 0)
        argmax_proba = np.argmax(probability_sum, axis = -1)
        
        diz_accuracy[', '.join(elements)] = accuracy_score(y_val,argmax_proba)
        diz_metrics[', '.join(elements)] =  precision_recall_fscore_support(y_val,argmax_proba, average=None)
    df_results = pd.DataFrame.from_dict(diz_accuracy, orient='index').rename({0: 'Accuracy'}, axis = 1).sort_values(by = 'Accuracy', ascending = False)
    
    outputs['Weights'] = weights_reliability
    outputs['Scoring'] = df_results
    outputs['Metrics'] = diz_metrics #precision, recall, fscore
    
    return outputs
    
def inference_bma(model1, test, weights):
    '''
    Prediction on test set, passing one model at time
    '''
    model1_proba = model1.predict_proba(test)
    prediction_proba0 = np.expand_dims(model1_proba[:,0] * weights['F0_weights'],1)
    prediction_proba1 = np.expand_dims(model1_proba[:,1] * weights['F1_weights'],1)
    prediction_weighted = np.concatenate([prediction_proba0, prediction_proba1], axis = 1)
    return prediction_weighted