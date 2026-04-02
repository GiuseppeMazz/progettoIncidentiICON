import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
import joblib
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
import matplotlib
from Predizione import df_predizione_utente
import warnings
import logging
from sklearn.feature_selection import SelectFromModel

from sklearn.pipeline import Pipeline


matplotlib.use('TkAgg')   # serveper il funzionamento di...
from sklearn.model_selection import train_test_split
from pgmpy.estimators import BayesianEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def predict_bayes(model, test_model):
    infer = VariableElimination(model)
    model_nodes = set(model.nodes())

    y_pred = []
    for _, row in test_model.iterrows():
        row_dict = row.drop('risk').to_dict()
        evidence = {k: int(v) for k, v in row_dict.items() if k in model_nodes}
        query = infer.query(['risk'], evidence=evidence)
        y_pred.append(query.values.argmax())

    return y_pred


def kfold_bayes(df, target, continuous_cols, edges, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Usa pd.qcut direttamente per la stratificazione sull'intero dataset
    y_cat_full = pd.qcut(df[target], q=3, labels=['Low', 'Medium', 'High'])

    accs, precs, recs, f1s = [], [], [], []

    tutti_y_true = []
    tutti_y_pred = []

    for train_idx, test_idx in skf.split(df, y_cat_full):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        exclude = ['INCIDENTI MORTALI', 'MORTI', 'FERITI', target]

        X_train = train_df.drop(columns=exclude)
        X_test = test_df.drop(columns=exclude)

        y_train = train_df[target]
        y_test = test_df[target]

        # Ora discretizza_target viene chiamata correttamente con 2 argomenti
        y_train_cat, y_test_cat = discretizza_target(y_train, y_test)

        train_model, test_model = prepara_bayes(
            X_train, X_test, y_train_cat, y_test_cat, continuous_cols
        )

        model = fit_bayes(train_model)
        y_pred = predict_bayes(model, test_model)
        y_true = test_model['risk'].values

        tutti_y_true.extend(y_true)
        tutti_y_pred.extend(y_pred)

        accs.append(accuracy_score(y_true, y_pred))
        precs.append(precision_score(y_true, y_pred, average='macro', zero_division=0))
        recs.append(recall_score(y_true, y_pred, average='macro', zero_division=0))
        f1s.append(f1_score(y_true, y_pred, average='macro', zero_division=0))

    print("\n=== BAYESIAN NETWORK ===")
    print(f"Accuracy: {np.mean(accs):.3f}")
    print(f"Precision: {np.mean(precs):.3f}")
    print(f"Recall: {np.mean(recs):.3f}")
    print(f"F1 Score:  {np.mean(f1s):.3f}")

    # distribuzione classi predette vs reali
    label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    tutti_y_true_label = [label_map[v] for v in tutti_y_true]
    tutti_y_pred_label = [label_map[v] for v in tutti_y_pred]

    print("\n=== DISTRIBUZIONE CLASSI ===")
    print(f"{'Classe':10s} {'Reali':>8s} {'%Reali':>8s} {'Predette':>10s} {'%Pred':>8s}")
    print("-" * 48)
    for label in ['Low', 'Medium', 'High']:
        n_true = tutti_y_true_label.count(label)
        n_pred = tutti_y_pred_label.count(label)
        pct_true = n_true / len(tutti_y_true_label) * 100
        pct_pred = n_pred / len(tutti_y_pred_label) * 100
        print(f"{label:10s} {n_true:>8d} {pct_true:>7.1f}% {n_pred:>10d} {pct_pred:>7.1f}%")

def fit_bayes(train_model):
    edges = [
        ('limiteVel', 'numeroCorsie'),
        ('limiteVel', 'mobility'),
        ('numeroCorsie', 'mobility'),

        ('poi_density', 'urban_density'),
        ('urban_density', 'complexity_index'),

        ('mobility', 'complexity_index'),
        ('complexity_index', 'attrito_stradale'),

        ('attrito_stradale', 'risk'),
        ('mobility', 'risk'),
        ('vulnerabilita', 'risk')
    ]

    model = DiscreteBayesianNetwork(edges)

    model.fit(
        train_model,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=10
    )

    return model

def kfold_classification(df, target, model, nome, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    bins = np.quantile(df[target], [0, 0.33, 0.66, 1.0])
    bins_ext = [-np.inf] + bins[1:-1].tolist() + [np.inf]
    labels = ['Low', 'Medium', 'High']
    y_cat_full=pd.cut(df[target], bins=bins_ext, labels=labels)

    accs, precs, recs = [], [], []

    for train_idx, test_idx in skf.split(df, y_cat_full):

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # ===== PREPARA DATI (no split) =====
        exclude = ['INCIDENTI MORTALI', 'MORTI', 'FERITI', target]

        X_train = train_df.drop(columns=exclude)
        y_train = train_df[target]

        X_test = test_df.drop(columns=exclude)
        y_test = test_df[target]

        # discretizzazione target
        bins = np.quantile(y_train, [0, 0.33, 0.66, 1.0])
        bins_ext = [-np.inf] + bins[1:-1].tolist() + [np.inf]
        labels = ['Low', 'Medium', 'High']

        y_train_cat = pd.cut(y_train, bins=bins_ext, labels=labels)
        y_test_cat = pd.cut(y_test, bins=bins_ext, labels=labels)

        # ===== MODELLO =====
        model.fit(X_train, y_train_cat)
        y_pred = model.predict(X_test)

        accs.append(accuracy_score(y_test_cat, y_pred))
        precs.append(precision_score(y_test_cat, y_pred, average='macro', zero_division=0))
        recs.append(recall_score(y_test_cat, y_pred, average='macro', zero_division=0))

    print(f"\n=== {nome} ===")
    print(f"Accuracy media: {np.mean(accs):.3f}")
    print(f"Precision media: {np.mean(precs):.3f}")
    print(f"Recall medio: {np.mean(recs):.3f}")


def modelli_class(X_train, X_test, y_train_cat, y_test_cat, df):
    soglie = ['mean', 'median', '1.25*mean']
    fs = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))

    rf_pipeline  = Pipeline([('fs', fs), ('clf', RandomForestClassifier(random_state=42))])
    gb_pipeline  = Pipeline([('fs', fs), ('clf', GradientBoostingClassifier())])
    svm_pipeline = Pipeline([('fs', fs), ('clf', SVC(kernel='rbf'))])
    knn_pipeline = Pipeline([('fs', fs), ('clf', KNeighborsClassifier())])

    kfold_classification(df, 'incidenti_per_km', rf_pipeline,  "RANDOM FOREST")
    kfold_classification(df, 'incidenti_per_km', gb_pipeline,  "GRADIENT BOOSTING")
    kfold_classification(df, 'incidenti_per_km', svm_pipeline, "SVC")
    kfold_classification(df, 'incidenti_per_km', knn_pipeline, "KNN")


def prepara_bayes(X_train, X_test, y_train_cat, y_test_cat, continuous_cols, n_bins=3):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    X_train_disc = X_train.copy()
    X_test_disc  = X_test.copy()

    X_train_disc[continuous_cols] = discretizer.fit_transform(X_train[continuous_cols])
    X_test_disc[continuous_cols]  = discretizer.transform(X_test[continuous_cols])
    train_model = X_train_disc.copy()
    test_model  = X_test_disc.copy()

    mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    train_model['risk'] = y_train_cat.astype(str).map(mapping)
    test_model['risk']  = y_test_cat.astype(str).map(mapping)

    # mobility
    train_model['mobility'] = train_model['limiteVel'] + train_model['numeroCorsie']
    test_model['mobility']  = test_model['limiteVel']  + test_model['numeroCorsie']

    mob_disc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    train_model[['mobility']] = mob_disc.fit_transform(train_model[['mobility']])
    test_model[['mobility']]  = mob_disc.transform(test_model[['mobility']])

    for col in train_model.columns:
        if col != 'risk':
            train_model[col] = train_model[col].astype(float).astype(int)
            test_model[col]  = test_model[col].astype(float).astype(int)

    return train_model, test_model


def prepara_dati(df, target):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    exclude = ['INCIDENTI MORTALI', 'MORTI', 'FERITI', target]

    X_train = train_df.drop(columns=exclude)
    y_train = train_df[target]

    X_test = test_df.drop(columns=exclude)
    y_test = test_df[target]

    # discretizzazione target
    bins = np.quantile(y_train, [0, 0.33, 0.66, 1.0])
    bins_extended = [-np.inf] + bins[1:-1].tolist() + [np.inf]
    labels = ['Low', 'Medium', 'High']
    y_train_cat = pd.cut(y_train, bins=bins_extended, labels=labels, include_lowest=True)
    y_test_cat = pd.cut(y_test, bins=bins_extended, labels=labels, include_lowest=True)

    return X_train, X_test, y_train, y_test, y_train_cat, y_test_cat


def discretizza_target(y_train, y_test):
    bins = np.quantile(y_train, [0, 0.33, 0.66, 1.0])
    bins_ext = [-np.inf] + bins[1:-1].tolist() + [np.inf]
    labels = ['Low', 'Medium', 'High']

    y_train_cat = pd.cut(y_train, bins=bins_ext, labels=labels, include_lowest=True)
    y_test_cat = pd.cut(y_test, bins=bins_ext, labels=labels, include_lowest=True)

    return y_train_cat, y_test_cat


def kfold_bayes_gridsearch(df, target, edges_list, continuous_cols, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_all_cat = pd.qcut(df[target], q=3, labels=['Low', 'Medium', 'High'])

    equivalent_sample_sizes = [1, 5, 10, 50]
    n_bins_options = [2, 3, 4]

    results = []

    for edges in edges_list:
        for ess in equivalent_sample_sizes:
            for n_bins in n_bins_options:

                accs, precs, recs, f1s = [], [], [], []

                for train_idx, test_idx in skf.split(df, y_all_cat):
                    train_df = df.iloc[train_idx]
                    test_df  = df.iloc[test_idx]

                    exclude = ['INCIDENTI MORTALI', 'MORTI', 'FERITI', target]
                    X_train = train_df.drop(columns=exclude)
                    y_train = train_df[target]
                    X_test  = test_df.drop(columns=exclude)
                    y_test  = test_df[target]

                    y_train_cat, y_test_cat = discretizza_target(y_train, y_test)

                    train_model, test_model = prepara_bayes(
                        X_train, X_test, y_train_cat, y_test_cat,
                        continuous_cols, n_bins=n_bins
                    )

                    try:
                        model = DiscreteBayesianNetwork(edges)
                        model.fit(
                            train_model,
                            estimator=BayesianEstimator,
                            prior_type="BDeu",
                            equivalent_sample_size=ess
                        )
                    except Exception as e:
                        accs.append(0); precs.append(0); recs.append(0); f1s.append(0)
                        continue

                    infer = VariableElimination(model)
                    model_nodes = set(model.nodes())

                    y_pred = []
                    for _, row in test_model.iterrows():
                        evidence = {
                            k: int(v)
                            for k, v in row.drop('risk').to_dict().items()
                            if k in model_nodes
                        }
                        query = infer.query(['risk'], evidence=evidence)
                        y_pred.append(query.values.argmax())

                    y_true = test_model['risk'].values
                    accs.append(accuracy_score(y_true, y_pred))
                    precs.append(precision_score(y_true, y_pred, average='macro', zero_division=0))
                    recs.append(recall_score(y_true, y_pred, average='macro', zero_division=0))
                    f1s.append(f1_score(y_true, y_pred, average='macro', zero_division=0))

                results.append({
                    "edges":     edges,
                    "ess":       ess,
                    "n_bins":    n_bins,
                    "n_edges":   len(edges),
                    "accuracy":  np.mean(accs),
                    "precision": np.mean(precs),
                    "recall":    np.mean(recs),
                    "f1":        np.mean(f1s)
                })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("f1", ascending=False).reset_index(drop=True)
    df_results[["accuracy", "precision", "recall", "f1"]] = \
        df_results[["accuracy", "precision", "recall", "f1"]].round(3)
    return df_results

def grid_src_classificazione_modelli(X_train, y_train_cat):
    soglie = ['mean', 'median', '1.25*mean']
    fs = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))

    scoring = {
        'accuracy':  'accuracy',
        'precision': 'precision_macro',
        'recall':    'recall_macro',
        'f1':        'f1_macro'
    }

    def stampa_risultati(nome, grid):
        idx = grid.best_index_
        cv  = grid.cv_results_
        print(f"\n=== {nome} ===")
        print(f"  Soglia FS:     {grid.best_params_['fs__threshold']}")
        print(f"  Best params:   { {k: v for k, v in grid.best_params_.items() if k != 'fs__threshold'} }")
        print(f"  Accuracy:      {cv['mean_test_accuracy'][idx]:.3f}")
        print(f"  Precision:     {cv['mean_test_precision'][idx]:.3f}")
        print(f"  Recall:        {cv['mean_test_recall'][idx]:.3f}")
        print(f"  F1 macro:      {cv['mean_test_f1'][idx]:.3f}")

    # RANDOM FOREST
    pipeline_rf = Pipeline([('fs', fs), ('clf', RandomForestClassifier(random_state=42))])
    param_grid_rf = {
        'fs__threshold': soglie,
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [None, 5, 10],
        'clf__min_samples_split': [2, 5]
    }
    gridRf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, scoring=scoring, refit='f1', n_jobs=-1)
    gridRf.fit(X_train, y_train_cat)
    stampa_risultati("RANDOM FOREST", gridRf)

    # GRADIENT BOOSTING
    pipeline_gb = Pipeline([('fs', fs), ('clf', GradientBoostingClassifier())])
    param_grid_gb = {
        'fs__threshold': soglie,
        'clf__n_estimators': [50, 100, 200],
        'clf__learning_rate': [0.01, 0.1, 0.2],
        'clf__max_depth': [3, 5, 7],
        'clf__min_samples_split': [2, 5],
        'clf__min_samples_leaf': [1, 3]
    }
    gridGb = GridSearchCV(pipeline_gb, param_grid_gb, cv=5, scoring=scoring, refit='f1', n_jobs=-1)
    gridGb.fit(X_train, y_train_cat)
    stampa_risultati("GRADIENT BOOSTING", gridGb)

    # SVC
    pipeline_svc = Pipeline([('fs', fs), ('clf', SVC())])
    param_grid_svc = [
        {
            'fs__threshold': soglie,
            'clf__kernel': ['linear'],
            'clf__C': [0.1, 1, 10, 100]
        },
        {
            'fs__threshold': soglie,
            'clf__kernel': ['rbf'],
            'clf__C': [0.1, 1, 10],
            'clf__gamma': ['scale', 0.01, 0.1, 1]
        }
    ]
    gridSvc = GridSearchCV(pipeline_svc, param_grid_svc, cv=5, scoring=scoring, refit='f1', n_jobs=-1)
    gridSvc.fit(X_train, y_train_cat)
    stampa_risultati("SVC", gridSvc)

    # KNN
    pipeline_knn = Pipeline([('fs', fs), ('clf', KNeighborsClassifier())])
    param_grid_knn = {
        'fs__threshold': soglie,
        'clf__n_neighbors': [3, 5, 7, 11, 15],
        'clf__weights': ['uniform', 'distance'],
        'clf__p': [1, 2]
    }
    gridKnn = GridSearchCV(pipeline_knn, param_grid_knn, cv=5, scoring=scoring, refit='f1', n_jobs=-1)
    gridKnn.fit(X_train, y_train_cat)
    stampa_risultati("KNN", gridKnn)

#####
### MENU' di NAVIGAZIONE
#####
def lavora_con_classificazione(df):

    #per warnings riguardanti la discretizzazione dei dati
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    #nascondere il log di pgmpy
    logging.getLogger("pgmpy").setLevel(logging.WARNING)

    risposta = True
    while risposta:
        print("\n")
        print('1) Indietro')
        print('2) prova modelli base')
        print('3) rete bayessiana')
        print('4) feature selection -> hyper param tuning ')
        print('5) grid search con rete bayes')
        print('6) salva modello rete')
        print('7) predizione con rete')
        print('8) distribuzione target')

        inputUtente = input("\nScelta (1-2-3-4-5): ").strip()
        match inputUtente:
            case '1':
                print("chiusura")
                risposta = False
            case '2':
                X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = prepara_dati(df, target = "incidenti_per_km")
                modelli_class(X_train, X_test, y_train_cat, y_test_cat,df)

            case '3':
                edges = [
                    # mobility
                    ('limiteVel', 'numeroCorsie'),
                    ('limiteVel', 'mobility'),
                    ('numeroCorsie', 'mobility'),
                    # urbano
                    ('poi_density', 'urban_density'),
                    ('urban_density', 'complexity_index'),
                    # propagazione
                    ('mobility', 'complexity_index'),
                    ('complexity_index', 'attrito_stradale'),
                    # rischio
                    ('attrito_stradale', 'risk'),
                    ('mobility', 'risk'),
                    # ('complexity_index', 'risk'),
                    ('vulnerabilita', 'risk')
                ]
                continuous_cols = [
                    'limiteVel', 'numeroCorsie', 'lunghezza_m',
                    'motorway_exposure', 'trunk_exposure', 'primary_exposure',
                    'complexity_index', 'junction_density', 'poi_totali',
                    'poi_density', 'urban_density', 'attrito_stradale',
                    'frammentazione', 'vulnerabilita',
                    # aggiunte perché usate in alcune edges_list
                    'strutture_scuole', 'strutture_ristoranti', 'strutture_supermercati',
                    'highway_motorway', 'highway_trunk', 'highway_primary']

                kfold_bayes(df, 'incidenti_per_km', continuous_cols , edges)

            case '4':

                X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = prepara_dati(df, target="incidenti_per_km")
                target = "incidenti_per_km"
                y_all_cat = pd.qcut(df[target], q=3, labels=['Low', 'Medium', 'High'])

                grid_src_classificazione_modelli(X_train, y_train_cat)

            case '5':
                print("ATTENDERE qualche minuto")
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                pd.set_option('display.max_colwidth', None)

                edges =[
                    # mobility
                    ('limiteVel', 'numeroCorsie'),
                    ('limiteVel', 'mobility'),
                    ('numeroCorsie', 'mobility'),
                    # urbano
                    ('poi_density', 'urban_density'),
                    ('urban_density', 'complexity_index'),
                    # propagazione
                    ('mobility', 'complexity_index'),
                    ('complexity_index', 'attrito_stradale'),
                    # rischio
                    ('attrito_stradale', 'risk'),
                    ('mobility', 'risk'),
                    # ('complexity_index', 'risk'),
                    ('vulnerabilita', 'risk')
                ]

                edges_list= [edges,[
                    ('limiteVel', 'numeroCorsie'),
                    ('limiteVel', 'mobility'),
                    ('numeroCorsie', 'mobility'),
                    # urbano
                    ('poi_density', 'urban_density'),
                    ('urban_density', 'complexity_index'),
                    # propagazione
                    ('mobility', 'complexity_index'),
                    #('complexity_index', 'attrito_stradale'),
                    # rischio
                    ('attrito_stradale', 'risk'),
                    ('complexity_index', 'risk'),
                    ('vulnerabilita', 'risk')
                ],[
                    # urbanizzazione -> attività
                    ('strutture_scuole', 'poi_density'),
                    ('strutture_ristoranti', 'poi_density'),
                    ('strutture_supermercati', 'poi_density'),
                    ('poi_density', 'urban_density'),
                    # urban -> traffico
                    ('urban_density', 'mobility'),
                    # infrastruttura -> traffico
                    ('limiteVel', 'mobility'),
                    ('numeroCorsie', 'mobility'),
                    # rete stradale -> complessità
                    ('junction_density', 'complexity_index'),
                    ('frammentazione', 'complexity_index'),
                    # traffico -> complessità
                    ('mobility', 'complexity_index'),
                    # complessità -> condizioni guida
                    ('complexity_index', 'risk'),
                    # rischio finale
                    ('attrito_stradale', 'risk'),
                    ('vulnerabilita', 'risk')
                ],[
                    # tipo strada -> exposure
                    ('highway_motorway', 'motorway_exposure'),
                    ('highway_trunk', 'trunk_exposure'),
                    ('highway_primary', 'primary_exposure'),
                    # exposure -> velocità
                    ('motorway_exposure', 'limiteVel'),
                    ('trunk_exposure', 'limiteVel'),
                    ('primary_exposure', 'limiteVel'),
                    # velocità -> mobilità
                    ('limiteVel', 'mobility'),
                    ('numeroCorsie', 'mobility'),
                    # mobilità -> rischio
                    ('mobility', 'risk'),
                    # condizioni strada
                    ('attrito_stradale', 'risk'),
                    # vulnerabilità
                    ('vulnerabilita', 'risk')
                ],[
                    # infrastruttura -> mobilità
                    ('limiteVel', 'numeroCorsie'),
                    ('limiteVel', 'mobility'),
                    ('numeroCorsie', 'mobility'),
                    # urbano -> complexity
                    ('poi_density', 'urban_density'),
                    ('urban_density', 'complexity_index'),
                    # rete stradale -> complexity
                    ('junction_density', 'complexity_index'),
                    ('frammentazione', 'complexity_index'),
                    # mobilità -> complexity
                    ('mobility', 'complexity_index'),
                    # complexity -> condizioni guida
                    ('complexity_index', 'attrito_stradale'),
                    # rischio finale
                    ('attrito_stradale', 'risk'),
                    ('mobility', 'risk'),
                    ('vulnerabilita', 'risk'),
                ]]

                continuous_cols = ['limiteVel', 'numeroCorsie', 'lunghezza_m',
                                    'motorway_exposure', 'trunk_exposure', 'primary_exposure',
                                    'complexity_index', 'junction_density', 'poi_totali',
                                    'poi_density', 'urban_density', 'attrito_stradale',
                                    'frammentazione', 'vulnerabilita',
                                    # aggiunte perché usate in alcune edges_list
                                    'strutture_scuole', 'strutture_ristoranti', 'strutture_supermercati',
                                    'highway_motorway', 'highway_trunk', 'highway_primary',]

                risultati = kfold_bayes_gridsearch(df, "incidenti_per_km", edges_list, continuous_cols, n_splits=5)

                risultati["n_edges"] = risultati["edges"].apply(len)
                # arrotonda metriche
                risultati["n_edges"] = risultati["edges"].apply(len)
                risultati["accuracy"] = risultati["accuracy"].round(3)
                risultati["precision"] = risultati["precision"].round(3)
                risultati["recall"] = risultati["recall"].round(3)
                risultati["f1"] = risultati["f1"].round(3)

                #ordina per accuracy
                risultati = risultati.sort_values(by="accuracy", ascending=False).reset_index(drop=True)

                # etichetta leggibile per ogni struttura di edges
                edges_labels = {i: f"struttura_{i + 1}" for i in range(len(edges_list))}
                risultati["edges_id"] = risultati["edges"].apply(
                    lambda e: next((f"struttura_{i + 1}" for i, el in enumerate(edges_list) if el == e), "?")
                )

                print("\n=== RISULTATI BAYES (ORDINATI PER ACCURACY) ===")
                print(risultati[["edges_id", "n_edges", "ess", "n_bins", "accuracy", "precision", "recall","f1"]].to_string(
                    index=True))

                print("\n=== ANALISI PER STRUTTURA EDGES ===")
                for edges_id, gruppo in risultati.groupby("edges_id"):
                    print(f"\n{edges_id} ({gruppo['n_edges'].iloc[0]} edges):")
                    print(
                        f"  accuracy  -> min: {gruppo['accuracy'].min():.3f}  max: {gruppo['accuracy'].max():.3f}  media: {gruppo['accuracy'].mean():.3f}")
                    print(f"  best ess  -> {gruppo.loc[gruppo['accuracy'].idxmax(), 'ess']}")
                    print(f"  best bins -> {gruppo.loc[gruppo['accuracy'].idxmax(), 'n_bins']}")

                print("\n=== MIGLIORE CONFIGURAZIONE ===")
                best = risultati.iloc[0]
                print(f"  edges:     {best['edges_id']}  ({best['n_edges']} edges)")
                print(f"  ess:       {best['ess']}")
                print(f"  n_bins:    {best['n_bins']}")
                print(f"  accuracy:  {best['accuracy']}")
                print(f"  precision: {best['precision']}")
                print(f"  recall:    {best['recall']}")
                print(f"  f1:        {best['f1']}")

            case '6':

                edges = [
                    # mobility
                    ('limiteVel', 'numeroCorsie'),
                    ('limiteVel', 'mobility'),
                    ('numeroCorsie', 'mobility'),

                    # urbano
                    ('poi_density', 'urban_density'),
                    ('urban_density', 'complexity_index'),

                    # propagazione
                    ('mobility', 'complexity_index'),
                    ('complexity_index', 'attrito_stradale'),

                    # rischio
                    ('attrito_stradale', 'risk'),
                    ('mobility', 'risk'),
                    # ('complexity_index', 'risk'),
                    ('vulnerabilita', 'risk')
                ]
                continuous_cols = [
                    'limiteVel', 'numeroCorsie',
                    'poi_density', 'urban_density',
                    'complexity_index', 'attrito_stradale',
                    'vulnerabilita'
                ]

                target = "incidenti_per_km"
                exclude = ['INCIDENTI MORTALI', 'MORTI', 'FERITI', target]

                X = df.drop(columns=exclude)
                y = df[target]

                # discretizzazione target
                bins = np.quantile(y, [0, 0.33, 0.66, 1.0])
                bins_ext = [-np.inf] + bins[1:-1].tolist() + [np.inf]
                labels = ['Low', 'Medium', 'High']
                y_cat = pd.cut(y, bins=bins_ext, labels=labels)


                discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
                discretizer.fit(X[continuous_cols])
                X_disc = X.copy()
                X_disc[continuous_cols] = discretizer.transform(X[continuous_cols])
                train_model = X_disc.copy()
                mapping = {'Low': 0, 'Medium': 1, 'High': 2}
                train_model['risk'] = y_cat.astype(str).map(mapping)
                # mobility (come in prepara_bayes)
                train_model['mobility'] = train_model['limiteVel'] + train_model['numeroCorsie']
                mob_disc = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
                mob_disc.fit(train_model[['mobility']])
                train_model[['mobility']] = mob_disc.transform(train_model[['mobility']])

                for col in train_model.columns:
                    if col != 'risk':
                        train_model[col] = train_model[col].astype('category').cat.codes

                ####################
                # preprocessing

                #train_model, _ = prepara_bayes(X, X, y_cat, y_cat, continuous_cols)

                # modello
                model = DiscreteBayesianNetwork(edges)
                model.fit(
                    train_model,
                    estimator=BayesianEstimator,
                    prior_type="BDeu",
                    equivalent_sample_size=10
                )

                joblib.dump({
                    "model": model,
                    "discretizer": discretizer,
                    "mob_disc": mob_disc,
                    "continuous_cols": continuous_cols
                }, "classificazione/bayesian_model.pkl")
                print("Modello salvato correttamente")

            case '7':

                if os.path.exists("classificazione/bayesian_model.pkl"):
                    print("File trovato")

                    data = joblib.load("classificazione/bayesian_model.pkl")
                    model = data["model"]
                    discretizer = data["discretizer"]  # oggetto fitted
                    mob_disc = data["mob_disc"]  # oggetto fitted
                    continuous_cols = data["continuous_cols"]

                    risposta_pred = True
                    while risposta_pred:
                        print("\n")
                        print("1) indietro")
                        print("2) dai in input Milano - Bergamo  (45.4642, 9.1910), (45.7040, 9.6630) [richiede 3 min. circa]")
                        print("3) dai in input Bari - Bisceglie (41.1108, 16.8741), (41.2373, 16.5030) [richiede 2 min. circa]")
                        print("4) dai in input Palermo - Messina  (38.0912, 13.3623), (38.1895, 15.5469) [richiede 30 min. circa]")
                        print("5) dai in input Arezzo - Viterbo (43.4582, 11.8899), (42.4507, 12.0890) [richiede 7 min. circa]")
                        print("6) dai in input Savona - Genova (44.3189, 8.4845), (44.4197, 8.9061) [richiede 2 min. circa]")
                        print("7) dai in input Avezzano - Torano di Borgorose (42.04695, 13.407336), (42.149964, 13.256215)")

                        print("\nATTENZIONE: ovviamente punti molto distanti possono chiedere abbastanza tempo")
                        inputUtente = input("\nScelta (1-2-3-4-5-6): ").strip()

                        print("Modello caricato correttamente")

                        match inputUtente:
                            case '1':
                                print("Indietro")
                                risposta_pred = False

                            case '2':
                                effettua_predizione_bayes("Milano - Bergamo", (45.4642, 9.1910), (45.7040, 9.6630),
                                                    model, discretizer, mob_disc, continuous_cols)
                            case '3':
                                effettua_predizione_bayes("Bari - Bisceglie", (41.1108, 16.8741), (41.237393, 16.5030),
                                                    model, discretizer, mob_disc, continuous_cols)

                            case '4':
                                effettua_predizione_bayes("Palermo - Messina", (38.0912, 13.3623), (38.1895, 15.5469),
                                                    model, discretizer, mob_disc, continuous_cols)

                            case '5':
                                effettua_predizione_bayes("Arezzo - Viterbo", (43.4582, 11.8899), (42.4507, 12.0890),
                                                    model, discretizer, mob_disc, continuous_cols)

                            case '6':
                                effettua_predizione_bayes("Savona - Genova", (44.3189, 8.4845), (44.4197, 8.9061),
                                                    model, discretizer, mob_disc, continuous_cols)
                            case '7':
                                effettua_predizione_bayes("Avezzano - Torano di Borgorose", (42.04695, 13.407336), (42.149964, 13.256215),
                                                          model, discretizer, mob_disc, continuous_cols)

                            case _:
                                print("input sbagliato, ritenta")

                else:
                    print("File NON trovato")

            case '8':
                #informazioni per la documentazione
                print(df['incidenti_per_km'].describe())
                print("*"*40)
                bins = np.quantile(df['incidenti_per_km'], [0, 0.33, 0.66, 1.0])
                bins_ext = [-np.inf] + bins[1:-1].tolist() + [np.inf]
                y_cat = pd.cut(df['incidenti_per_km'], bins=bins_ext, labels=['Low', 'Medium', 'High'])
                print(y_cat.value_counts().sort_index())
                print(bins)

            case _:
                print("input sbagliato, ritenta")




#####
### Predizione con la RETE Bayessiana
#####

def effettua_predizione_bayes(nome, start, end, model, discretizer, mob_disc, continuous_cols):

    df_raw = df_predizione_utente(nome, start, end)

    infer = VariableElimination(model)

    df_disc = df_raw.copy()
    df_disc[continuous_cols] = discretizer.transform(df_raw[continuous_cols])

    df_disc['mobility'] = df_disc['limiteVel'] + df_disc['numeroCorsie']
    df_disc[['mobility']] = mob_disc.transform(df_disc[['mobility']])

    for col in df_disc.columns:
        df_disc[col] = df_disc[col].astype(float).astype(int)

    predictions = []
    for _, row in df_disc.iterrows():
        evidence = build_evidence(row, model.nodes())
        query = infer.query(['risk'], evidence=evidence)
        pred = int(query.values.argmax())
        predictions.append(pred)

    label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    print(f"\n=== Predizioni per {nome} ===")
    for i, p in enumerate(predictions):
        print(f"  Segmento {i+1}: {label_map.get(p, p)}")


"""
def compute_bins(df, continuous_cols):
    bins_dict = {}
    for col in continuous_cols:
        q = np.quantile(df[col], [0, 0.33, 0.66, 1.0])
        bins = [-np.inf] + q[1:-1].tolist() + [np.inf]
        bins_dict[col] = bins
    return bins_dict


def discretize_new_data(df, bins_dict):
    df_disc = df.copy()
    for col, bins in bins_dict.items():
        # bins_dict[col] deve avere lo stesso numero di intervalli usato in training
        num_bins = len(bins) - 1
        df_disc[col] = pd.cut(df[col], bins=bins, labels=range(num_bins))
    return df_disc
"""

def build_evidence(row, model_nodes):
    evidence = {}
    for col in row.index:
        if col in model_nodes and col != 'risk':
            if pd.notna(row[col]):
                evidence[col] = int(row[col])
    return evidence