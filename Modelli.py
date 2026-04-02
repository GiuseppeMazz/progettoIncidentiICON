from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from ValutazioneModelli import k_fold_personalizzato, nested_kfold_grid_src, ottieni_prestazioni_modelli
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
import joblib
import os

from sklearn.feature_selection import SelectKBest, f_regression, SequentialFeatureSelector
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.feature_selection import SelectFromModel



def dividiDataset (df_appiattito, target = 'INCIDENTI'):
    y = df_appiattito[target]
    X = df_appiattito.drop([target, 'INCIDENTI MORTALI', 'MORTI', 'FERITI'], axis=1)


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



def grid_search_KNN(X_train, y_train):
    param = {
        'knn__n_neighbors': [3, 5, 7, 10, 15, 20],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan', 'minkowski'],
        'knn__p': [1, 2],  # rilevante solo per minkowski
    }
    pipe = Pipeline([
        ('scaler', StandardScaler()),  # KNN è sensibile alla scala
        ('knn', KNeighborsRegressor())
    ])
    grid_search = GridSearchCV(
        pipe,
        param_grid=param,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train.values.ravel())
    return grid_search.best_estimator_


def grid_search_ElasticNet(X_train, y_train):
    param = {
        'elasticnet__alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        'elasticnet__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        'elasticnet__max_iter': [1000, 5000],
    }
    pipe = Pipeline([
        ('scaler', StandardScaler()),  # ElasticNet è sensibile alla scala
        ('elasticnet', ElasticNet(random_state=42))
    ])
    grid_search = GridSearchCV(
        pipe,
        param_grid=param,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train.values.ravel())
    return grid_search.best_estimator_


def grid_search_BayesianRidge(X_train, y_train):
    param = {
        'bayridge__alpha_1': [1e-6, 1e-5, 1e-4],
        'bayridge__alpha_2': [1e-6, 1e-5, 1e-4],
        'bayridge__lambda_1': [1e-6, 1e-5, 1e-4],
        'bayridge__lambda_2': [1e-6, 1e-5, 1e-4],
    }
    pipe = Pipeline([
        ('scaler', StandardScaler()),  # BayesianRidge è sensibile alla scala
        ('bayridge', BayesianRidge())
    ])
    grid_search = GridSearchCV(
        pipe,
        param_grid=param,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
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



#sequential feature selection, NOTA: una feature selection con Sequential feature selection richiede tanto tempo
# in quanto si provano tutte el combinazioni assieme
# quindi si è optato per una select k best con k incrementale
def feat_selection(model, X, y, min_features=8, step=1):
    print("Inizio feat selection")
    n_features_range = list(range(min_features, X.shape[1] + 1, step))
    risultati_mae = []
    risultati_mse = []
    risultati_r2  = []  # aggiunto

    for k in n_features_range:
        pipeline = Pipeline([
            ('select', SelectKBest(score_func=f_regression, k=k)),
            ('model', clone(model))
        ])

        risultati = k_fold_personalizzato(pipeline, X, y, nome=f"KBest_{k}")
        risultati_mae.append(risultati["mae_mean"])
        risultati_mse.append(risultati["mse_mean"])
        risultati_r2.append(risultati["r2_mean"])   # aggiunto

    # scelta basata su R² (massimizziamo, non minimizziamo)
    miglior_num_features_mae = n_features_range[np.argmin(risultati_mae)]
    miglior_num_features_mse = n_features_range[np.argmin(risultati_mse)]
    miglior_num_features_r2  = n_features_range[np.argmax(risultati_r2)]   # aggiunto

    # recupero feature selezionate in base a R²
    selector_r2 = SelectKBest(score_func=f_regression, k=miglior_num_features_r2)
    selector_r2.fit(X, y)
    feat_r2 = X.columns[selector_r2.get_support()].tolist()

    print(f"  Miglior R²:  {max(risultati_r2):.4f} con {miglior_num_features_r2} feature")
    print(f"  Miglior MAE: {min(risultati_mae):.4f} con {miglior_num_features_mae} feature")
    print(f"  Feature scelte (R²): {feat_r2}")

    return (n_features_range,
            risultati_mae,
            risultati_mse,
            risultati_r2,                  # aggiunto
            miglior_num_features_mae,
            miglior_num_features_mse,
            miglior_num_features_r2,
            feat_r2)                       # ora le feature sono scelte per R²



def gridSrc_con_featSel_integrata(df):
 # Approccio rigoroso: feature selection DENTRO la pipeline del nested kfold
    # → nessun data leakage, approccio raccomandato per la valutazione finale
    print("\nGridSearch con feature selection integrata nella pipeline")
    print("grid con Gradient Boosting")

    X_train, X_valid, y_train, y_valid, X, y = dividiDataset(df)

    # --- GRADIENT BOOSTING ---
    pipe_gb = Pipeline([
        ('feat_sel', SelectFromModel(
            GradientBoostingRegressor(n_estimators=100, random_state=42),
            threshold="median"
        )),
        ('model', GradientBoostingRegressor())
    ])
    param_gb = {
        'feat_sel__estimator__n_estimators': [100],
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.05, 0.1, 0.2],
        'model__max_depth': [3, 4, 5],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2, 4],
        'model__subsample': [0.8, 1.0],
        'model__max_features': ['sqrt', 'log2'],
        'model__loss': ['squared_error']
    }
    risultatiGb = nested_kfold_grid_src(pipe_gb, param_gb, X, y, "GB pipeline")
    print(f"\n{risultatiGb['nome']} → MAE: {risultatiGb['mae_mean']:.4f} (±{risultatiGb['mae_std']:.4f})"
          f" | R²: {risultatiGb['r2_mean']:.4f} (±{risultatiGb['r2_std']:.4f})")

    # --- RANDOM FOREST ---


    print("grid con random forest")
    pipe_rf = Pipeline([
        ('feat_sel', SelectFromModel(
            RandomForestRegressor(n_estimators=100, random_state=42),
            threshold="median"
        )),
        ('model', RandomForestRegressor())
    ])
    param_rf = {
        'feat_sel__estimator__n_estimators': [100, 200],
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt'],
        'model__bootstrap': [True]
    }
    risultatiRf = nested_kfold_grid_src(pipe_rf, param_rf, X, y, "RF pipeline")
    print(f"{risultatiRf['nome']} → MAE: {risultatiRf['mae_mean']:.4f} (±{risultatiRf['mae_std']:.4f})"
          f" | R²: {risultatiRf['r2_mean']:.4f} (±{risultatiRf['r2_std']:.4f})")

    # --- SVR ---


    print("grid con SVR")
    pipe_svr = Pipeline([
        ('scaler', StandardScaler()),
        ('feat_sel', SelectFromModel(SVR(kernel='linear'), threshold="median")),
        ('svr', SVR())
    ])
    param_svr = [{
        "svr__kernel": ["linear"],
        "svr__C": [0.1, 1, 10],
        "svr__epsilon": [0.1, 0.5]
    }, {
        "svr__kernel": ["rbf"],
        "svr__C": [1, 10, 100],
        "svr__epsilon": [0.1, 0.5],
        "svr__gamma": ["scale", 0.1]
    }]
    risultatiSvr = nested_kfold_grid_src(pipe_svr, param_svr, X, y, "SVR pipeline")
    print(f"{risultatiSvr['nome']} → MAE: {risultatiSvr['mae_mean']:.4f} (±{risultatiSvr['mae_std']:.4f})"
          f" | R²: {risultatiSvr['r2_mean']:.4f} (±{risultatiSvr['r2_std']:.4f})")

    # --- KNN ---
    print("grid KNN")
 # KNN non ha coef_ né feature_importances_, quindi SelectFromModel non è applicabile.
 # Si usa SelectKBest che è la scelta più neutra e corretta per modelli senza importanze
    pipe_knn = Pipeline([
        ('scaler', StandardScaler()),
        ('feat_sel', SelectKBest(score_func=f_regression, k='all')),  # k ottimizzato dalla grid search
        ('knn', KNeighborsRegressor())
    ])
    param_knn = {
        'feat_sel__k': [10, 15, 20, 25],  # k diventa iperparametro della grid search
        'knn__n_neighbors': [3, 5, 7, 10, 15],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan'],
    }
    risultatiKnn = nested_kfold_grid_src(pipe_knn, param_knn, X, y, "KNN pipeline")
    print(f"{risultatiKnn['nome']} → MAE: {risultatiKnn['mae_mean']:.4f} (±{risultatiKnn['mae_std']:.4f})"
          f" | R²: {risultatiKnn['r2_mean']:.4f} (±{risultatiKnn['r2_std']:.4f})")

    # --- ELASTICNET ---
    print("grid con elastic Net")
    pipe_en = Pipeline([
        ('scaler', StandardScaler()),
        ('feat_sel', SelectFromModel(ElasticNet(random_state=42), threshold="median")),  # coerente
        ('elasticnet', ElasticNet(random_state=42))
    ])
    param_en = {
        'feat_sel__estimator__alpha': [0.1, 1.0],  # iperparametro del selettore
        'elasticnet__alpha': [0.01, 0.1, 0.5, 1.0, 5.0],
        'elasticnet__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'elasticnet__max_iter': [1000, 5000],
    }
    risultatiEn = nested_kfold_grid_src(pipe_en, param_en, X, y, "ElasticNet pipeline")
    print(f"{risultatiEn['nome']} → MAE: {risultatiEn['mae_mean']:.4f} (±{risultatiEn['mae_std']:.4f})"
          f" | R²: {risultatiEn['r2_mean']:.4f} (±{risultatiEn['r2_std']:.4f})")

    # --- BAYESIAN RIDGE ---
    print("grid con bayessian ridge")
    pipe_br = Pipeline([
        ('scaler', StandardScaler()),
        ('feat_sel', SelectFromModel(BayesianRidge(), threshold="median")),  # coerente
        ('bayridge', BayesianRidge())
    ])
    param_br = {
        'bayridge__alpha_1': [1e-6, 1e-5, 1e-4],
        'bayridge__alpha_2': [1e-6, 1e-5, 1e-4],
        'bayridge__lambda_1': [1e-6, 1e-5, 1e-4],
        'bayridge__lambda_2': [1e-6, 1e-5, 1e-4],
    }
    risultatiBr = nested_kfold_grid_src(pipe_br, param_br, X, y, "BayesianRidge pipeline")
    print(f"{risultatiBr['nome']} → MAE: {risultatiBr['mae_mean']:.4f} (±{risultatiBr['mae_std']:.4f})"
          f" | R²: {risultatiBr['r2_mean']:.4f} (±{risultatiBr['r2_std']:.4f})")




def salva_modello_finale_GB(df):
    X_train, X_valid, y_train, y_valid, X, y = dividiDataset(df)

    # Step 1: feature selection solo su X_train
    selector = SelectFromModel(
        GradientBoostingRegressor(n_estimators=100, random_state=42),
        threshold="median"
    )
    selector.fit(X_train, y_train)
    FEATURE_SCELTE = X_train.columns[selector.get_support()].tolist()
    print(f"Feature selezionate ({len(FEATURE_SCELTE)}): {FEATURE_SCELTE}")

    X_train_sel = X_train[FEATURE_SCELTE]
    X_sel = X[FEATURE_SCELTE]

    # Step 2: grid search
    bestModel = grid_search_gradientBoosting(X_train_sel, y_train)

    # Step 3: valutazione
    risultati = k_fold_personalizzato(bestModel, X_sel, y, "GB Finale")
    ottieni_prestazioni_modelli([risultati])
    os.makedirs("modello", exist_ok=True)

    # Step 4: fit su tutto X e salva modello + feature insieme

    bestModel.fit(X_sel, y)
    joblib.dump({
        "modello": bestModel,
        "feature": FEATURE_SCELTE
    }, "modello/modelloGB_finale.joblib")
    print("Modello e feature salvati in 'modelloGB_finale.joblib'")
