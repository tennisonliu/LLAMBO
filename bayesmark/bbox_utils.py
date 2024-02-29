# NOTE: ALL SEEDING IS DONE IN LLAMBO.PY

from functools import partial

def get_bayesmark_func(model_name, task_type, y_train=None):
    # https://github.com/uber/bayesmark/blob/master/bayesmark/sklearn_funcs.py
    assert model_name in ['RandomForest', 'DecisionTree', 'SVM', 'AdaBoost', 'MLP_SGD'], f'Unknown model name: {model_name}'
    assert task_type in ['classification', 'regression']
    if model_name == 'RandomForest':
        if task_type == 'classification':
            from sklearn.ensemble import RandomForestClassifier
            return partial(RandomForestClassifier, n_estimators=10, max_leaf_nodes=None, random_state=0)    # following Bayesmark implementation
        elif task_type == 'regression':
            from sklearn.ensemble import RandomForestRegressor
            return partial(RandomForestRegressor, n_estimators=10, max_leaf_nodes=None, random_state=0)
        
    if model_name == 'DecisionTree':
        if task_type == 'classification':
            from sklearn.tree import DecisionTreeClassifier
            return partial(DecisionTreeClassifier, max_leaf_nodes=None, random_state=0)
        elif task_type == 'regression':
            from sklearn.tree import DecisionTreeRegressor
            return partial(DecisionTreeRegressor, max_leaf_nodes=None, random_state=0)

    if model_name == 'SVM':
        if task_type == 'classification':
            from sklearn.svm import SVC
            return partial(SVC, kernel='rbf', probability=True, random_state=0)
        elif task_type == 'regression':
            from sklearn.svm import SVR
            return partial(SVR, kernel='rbf')
        
    if model_name == 'AdaBoost':
        if task_type == 'classification':
            from sklearn.ensemble import AdaBoostClassifier
            return partial(AdaBoostClassifier, random_state=0)
        elif task_type == 'regression':
            from sklearn.ensemble import AdaBoostRegressor
            return partial(AdaBoostRegressor, random_state=0)
    
    if model_name == 'MLP_SGD':
        if task_type == 'classification':
            from sklearn.neural_network import MLPClassifier
            return partial(MLPClassifier, solver='sgd', early_stopping=True, max_iter=40,
                           learning_rate='invscaling', nesterovs_momentum=True, random_state=0)
        elif task_type == 'regression':
            from sklearn.neural_network import MLPRegressor
            return partial(MLPRegressor, solver='sgd', activation='tanh', early_stopping=True, max_iter=40,
                           learning_rate='invscaling', nesterovs_momentum=True, random_state=0)