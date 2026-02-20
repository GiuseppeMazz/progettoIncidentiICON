from sklearn.base import clone

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from ValutazioneModelli import k_fold_personalizzato
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
import joblib
import os


from sklearn.feature_selection import SelectKBest, f_regression, SequentialFeatureSelector
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import numpy as np



def dividiDataset (df_appiattito):
    y = df_appiattito['INCIDENTI']
    X = df_appiattito.drop(['INCIDENTI', 'INCIDENTI MORTALI', 'MORTI', 'FERITI'], axis=1)


    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                          random_state=0)
    return X_train, X_valid, y_train, y_valid, X, y


def trainModelli (X, y):
    risultati = []
    #l'SVR sarà  usato solo assieme ad uno scaler in quanto troppo sensibile
    # a feature con valori molto diversi

    pipeline_svr_scalato = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVR(kernel='linear'))
    ])
    risultati_svr_scalato = k_fold_personalizzato(pipeline_svr_scalato, X, y, "Svr_scalato")
    risultati.append(risultati_svr_scalato)

    modelloRandForest = RandomForestRegressor(random_state=42)
    risultati_randForest = k_fold_personalizzato(modelloRandForest, X, y, "Rand forest")
    risultati.append(risultati_randForest)

    knnr=KNeighborsRegressor(n_neighbors=5)
    risultati_knnr = k_fold_personalizzato(knnr, X, y, "Knnr")
    risultati.append(risultati_knnr)

    pipeline_knn_scalato = Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsRegressor(n_neighbors=5))
    ])

    risultati_knn_scalato = k_fold_personalizzato(pipeline_knn_scalato, X, y, "Knnr scalato")
    risultati.append(risultati_knn_scalato)

    gradBoost = GradientBoostingRegressor(random_state=42)
    risultati_gradBoost = k_fold_personalizzato(gradBoost, X, y, "Gradiant Boosting")
    risultati.append(risultati_gradBoost)

    pipeline_ElasticNet_scalato = Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNet(random_state=42))
    ])

    risultati_elasticNet = k_fold_personalizzato(pipeline_ElasticNet_scalato, X, y, "Elastic net scalato")
    risultati.append(risultati_elasticNet)

    bayes = BayesianRidge()
    risultati_bayes = k_fold_personalizzato(bayes, X, y, "Bayesian ridge")
    risultati.append(risultati_bayes)

    return risultati


def grid_search_randomForest(X_train, y_train):
    parameters_fast = {
        'randForest__n_estimators': [100, 200, 300],
        'randForest__max_depth': [None, 10, 20],
        'randForest__min_samples_split': [2, 5],
        'randForest__min_samples_leaf': [1, 2, 4],
        'randForest__max_features': ['sqrt'],
        'randForest__bootstrap': [True]
    }

    pipeRandForest = Pipeline([
        ('scaler', StandardScaler()),
        ('randForest', RandomForestRegressor(random_state=42))
    ])

    grid_search = GridSearchCV(
        pipeRandForest,
        param_grid=parameters_fast,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1)
    grid_search.fit(X_train, y_train.values.ravel())

    #restituisci il modello
    return grid_search.best_estimator_


def grid_search_SVR(X_train, y_train, modello=None):
    param = [{
        "svr__kernel": ["linear"],
        "svr__C": [0.1, 1, 10],
        "svr__epsilon": [0.1, 0.5]
    },{
        "svr__kernel": ["rbf"],
        "svr__C": [1, 10, 100],
        "svr__epsilon": [0.1, 0.5],
        "svr__gamma": ["scale", 0.1]
    }]

    if modello is None:
        pipeSvr = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR())
        ])
    else:
        pipeSvr = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', modello)
        ])

    grid_search= GridSearchCV(
        pipeSvr,
        param_grid = param,
        cv = 5,
        scoring = 'neg_mean_absolute_error',
        n_jobs = -1,
        verbose = 1)
    grid_search.fit(X_train, y_train.values.ravel())
    return grid_search.best_estimator_


def grid_search_gradientBoosting(X_train, y_train):
    # gridsearch gradientBoosting
    param = {
        'gradBoost__n_estimators': [100, 200, 300],
        'gradBoost__learning_rate': [0.05, 0.1, 0.2],
        'gradBoost__max_depth': [3, 4, 5],
        'gradBoost__min_samples_split': [2, 4, 6],
        'gradBoost__min_samples_leaf': [1, 2, 4],
        'gradBoost__subsample': [0.7, 0.85, 1.0],
        'gradBoost__max_features': ['sqrt', 'log2'],
        'gradBoost__loss': ['squared_error']
    }

    pipeGradBoost = Pipeline([

        ('gradBoost', GradientBoostingRegressor())
    ])

    grid_search = GridSearchCV(
        pipeGradBoost,
        param_grid=param,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,  # Parallelizza
        verbose=1)
    grid_search.fit(X_train, y_train.values.ravel())

    return grid_search.best_estimator_

def prestazioni_modelli_gridSearch(X_train, X_valid, y_train, y_valid, X, y):
    modello_gradBoost = grid_search_gradientBoosting(X_train, y_train)
    modello_randForest = grid_search_randomForest(X_train,y_train)
    modello_Svr = grid_search_SVR(X_train, y_train)

    risultati=[]
    risultati.append(k_fold_personalizzato(modello_gradBoost, X, y, "Gradient Boosing con Grid Src"))
    risultati.append(k_fold_personalizzato(modello_randForest, X, y, "RandForest con Grid Src"))
    risultati.append(k_fold_personalizzato(modello_Svr, X, y, "SVR con Grid Src"))
    return risultati

def salva_modello(model, nome_file):
    os.makedirs("modello", exist_ok=True)
    path = os.path.join("modello", nome_file)
    joblib.dump(model, path)
    print(f" Modello salvato in {path}")



#sequential feature selection, NOTA: una feature selection con Sequential feature selection richiede tanto tempo
# in quanto si provano tutte el combinazioni assieme
# quindi si è optato per una select k best con k incrementale

def feat_selection(model, X, y, min_features=8, step=1):
    print("Inizio feat selection")
    n_features_range = list(range(min_features, X.shape[1] + 1, step))
    risultati_mae = []
    risultati_mse = []

    for k in n_features_range:
        # Pipeline semplice: SelectKBest + modello
        pipeline = Pipeline([
            ('select', SelectKBest(score_func=f_regression, k=k)),
            ('model', clone(model))
        ])

        # Valuta con cross-validation
        risultati = k_fold_personalizzato(pipeline, X, y, nome=f"KBest_{k}")
        risultati_mae.append(risultati["mae_mean"])
        risultati_mse.append(risultati["mse_mean"])

    miglior_num_features_mae = n_features_range[np.argmin(risultati_mae)]
    miglior_num_features_mse = n_features_range[np.argmin(risultati_mse)]

    return (n_features_range, risultati_mae, risultati_mse,
            miglior_num_features_mae, miglior_num_features_mse)


def modello_finale(modello):
    pipeline = Pipeline([
        # k=26 ottenuto grazie ai test di feature selection
        ("kbest", SelectKBest(score_func=f_regression, k=26)),
        ("model", modello)
    ])
    return pipeline


