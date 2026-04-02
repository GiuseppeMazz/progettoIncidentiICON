from AnalizzaStrade import iqr_leggero
from Modelli import dividiDataset, trainModelli, grid_search_randomForest, grid_search_gradientBoosting, \
    grid_search_SVR, feat_selection
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from ValutazioneModelli import ottieni_prestazioni_modelli, k_fold_personalizzato, nested_kfold_grid_src, \
    ottieni_grafici_features_selection
from PreparaDataset import load_dataset
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor


from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor


#il seguente file è l'inizio di un secondo approcio al problema
# in cui il target non sarà più gli incidenti grezzi ma gli inidenti su km
# sono ancehe compresi i menu riguardanti il nuovo target



def cambia_target_evita_inf(df):
    df["incidenti_per_km"] = np.where(
        (df["lunghezza_m"]/1000) != 0,
        df["INCIDENTI"] / (df["lunghezza_m"]/1000),
        np.nan
    )

    df = df.dropna()
    df.drop(columns=["INCIDENTI"], inplace=True)
    return df

def modelli_gridSrc_target_nuovo():
    risposta_gs = True

    while risposta_gs:
        print("\n")
        print("1) torna indietro")
        print("2) applica GridSearch su Gradient Boosting (richiede abbastanza tempo)")
        print("3) applica GridSearch su Random Forest (ricihede un po' di tempo)")
        print("4) applica GridSearch su SVR ")
        inputUtente = input("\nScelta (1-2-3-4): ").strip()
        match inputUtente:
            case '1':
                print("Indietro")
                risposta_gs = False

            case '2':
                df = load_dataset()
                df = cambia_target_evita_inf(df)
                df, _ = iqr_leggero(df, target="incidenti_per_km")
                X_train, X_valid, y_train, y_valid, X, y = dividiDataset(df, "incidenti_per_km")
                bestModel = grid_search_gradientBoosting(X_train, y_train)
                risultatiGridGb = k_fold_personalizzato(bestModel, X, y, "Gradient Boosting con Grid Search New")
                ottieni_prestazioni_modelli([risultatiGridGb])

            case '3':
                df = load_dataset()
                df = cambia_target_evita_inf(df)
                df, _ = iqr_leggero(df, target="incidenti_per_km")
                X_train, X_valid, y_train, y_valid, X, y = dividiDataset(df, "incidenti_per_km")
                bestModel = grid_search_randomForest(X_train, y_train)
                risultatiGridRf = k_fold_personalizzato(bestModel, X, y, "Random Forest con Grid Search New")
                ottieni_prestazioni_modelli([risultatiGridRf])
            case '4':
                df = load_dataset()
                df = cambia_target_evita_inf(df)
                df, _ = iqr_leggero(df, target="incidenti_per_km")
                X_train, X_valid, y_train, y_valid, X, y = dividiDataset(df, "incidenti_per_km")
                bestModel = grid_search_SVR(X_train, y_train)
                risultatiGridSvr = k_fold_personalizzato(bestModel, X, y, "SVR con Grid Search New")
                ottieni_prestazioni_modelli([risultatiGridSvr])

            case _:
                print("input sbagliato, ritenta")



def lavora_con_target_nuovo(plt=None):
    risposta = True
    while risposta:
        print("\n")
        print("1) Indietro")
        print("2) training e risultati dei modelli base")
        print("3) controlla dataset")
        print("4) stampa dataset modificato")
        print("5) analisi errori")
        print("6) lavora con grid search")
        #print("7) esperimenti")
        print("7) gridsearch sulle feature scelte nella feature seletion")
        print("8) feature selection su nuovo target")
        #print("10) onologie")

        inputUtente = input("\nScelta (1-2-3-4): ").strip()
        match inputUtente:

            case '1':
                print("indietro")
                risposta = False

            case '2':
                df = load_dataset()
                #df = cambiaTarget(df)
                df = cambia_target_evita_inf(df)
                df, _ = iqr_leggero(df, target="incidenti_per_km")
                X_train, X_valid, y_train, y_valid, X, y = dividiDataset(df, "incidenti_per_km")

                risultati = trainModelli(X, y)
                ottieni_prestazioni_modelli(risultati)
            case '3':
                df = load_dataset()
                #df = cambiaTarget(df)
                df = cambia_target_evita_inf(df)
                df, _ = iqr_leggero(df, target="incidenti_per_km")
                X_train, X_valid, y_train, y_valid, X, y = dividiDataset(df, "incidenti_per_km")

                print("Inf in y:", np.isinf(y).sum())
                print("NaN in y:", np.isnan(y).sum())
                print("Media:", y.mean())
                print("Mediana:", y.median())
                print("Max:", y.max())
                print("Std:", y.std())

                pd.set_option('display.max_columns', None)
                print(df.describe().T)

            case '4':
                print('\n')
                df = load_dataset()
                #df = cambiaTarget(df)
                df = cambia_target_evita_inf(df)
                pd.set_option('display.max_columns', None)
                #iqr per toglier ei valori fuori dalla norma in incidenti_per_km
                # questo aiuta a eliminare dati errati che prima non erano facilmente visibili
                df,_ =iqr_leggero(df, target="incidenti_per_km")
                print(df.describe().T)
                
            case '5':
                df = load_dataset()
                # df = cambiaTarget(df)
                df = cambia_target_evita_inf(df)
                df, _ = iqr_leggero(df, target="incidenti_per_km")
                X_train, X_valid, y_train, y_valid, X, y = dividiDataset(df, "incidenti_per_km")
                #residui = y_test - y_pred
                #plt.scatter(y_pred, residui)
                plt.axhline(y=0, color='r', linestyle='--')

            case '6':
                modelli_gridSrc_target_nuovo()

            #case '7':
                #esperimenti
                # si vuole calcolare gli incidenti come si faceva periam ma facendo un ulteriorire IQR sul rapporto
                # incidenti strada perchè si è notato che in alcuni casi hanno dati praticamente da stragie,
                # qindi possono aver introdotto ,molto rmore nel modello
            #    esperimenti()

            case '7':
                print("\nGridSearch su feature selection")

                df = load_dataset()
                df = cambia_target_evita_inf(df)
                df, _ = iqr_leggero(df, target="incidenti_per_km")
                X_train, X_valid, y_train, y_valid, X, y = dividiDataset(df, "incidenti_per_km")
                print(f"target: {y.describe}")
                print(f'\nMedia del target "incidenti_per_km":{y.mean()}')
                print(f'Mediana del target "incidenti_per_km": {y.median()}')

                # --- GRADIENT BOOSTING con feature selection nella pipeline ---
                pipe_gb = Pipeline([
                    ('feat_sel',
                     SelectFromModel(GradientBoostingRegressor(n_estimators=100, random_state=42), threshold="median")),
                    ('model', GradientBoostingRegressor())
                ])
                param_Gb = {
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
                risultatiGb = nested_kfold_grid_src(pipe_gb, param_Gb, X, y, "GB")
                print(f"{risultatiGb['nome']} ha mae:{risultatiGb['mae_mean']}, con std: {risultatiGb['mae_std']}")
                print(f"mse: {risultatiGb['mse_mean']}, con std: {risultatiGb['mse_std']}")
                print(f"R^2: {risultatiGb['r2_mean']}, con std: {risultatiGb['r2_std']}")

                # --- RANDOM FOREST con feature selection nella pipeline ---
                pipe_rf = Pipeline([
                    ('feat_sel',
                     SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42), threshold="median")),
                    ('model', RandomForestRegressor())
                ])
                paramForest = {
                    'feat_sel__estimator__n_estimators': [100, 200],
                    'model__n_estimators': [100, 200, 300],
                    'model__max_depth': [None, 10, 20],
                    'model__min_samples_split': [2, 5],
                    'model__min_samples_leaf': [1, 2, 4],
                    'model__max_features': ['sqrt'],
                    'model__bootstrap': [True]
                }
                risultatiRf = nested_kfold_grid_src(pipe_rf, paramForest, X, y, "RF")
                print(f"{risultatiRf['nome']} ha mae:{risultatiRf['mae_mean']}, con std: {risultatiRf['mae_std']}")
                print(f"mse: {risultatiRf['mse_mean']}, con std: {risultatiRf['mse_std']}")
                print(f"R^2: {risultatiRf['r2_mean']}, con std: {risultatiRf['r2_std']}")

                # --- SVR con scaler + feature selection nella pipeline ---
                pipe_svr = Pipeline([
                    ('scaler', StandardScaler()),
                    ('feat_sel', SelectFromModel(
                        SVR(kernel='linear'),  # usa un LinearSVR per la feature selection (più efficiente)
                        threshold="median"
                    )),
                    ('svr', SVR())
                ])
                paramSvr = [{
                    "svr__kernel": ["linear"],
                    "svr__C": [0.1, 1, 10],
                    "svr__epsilon": [0.1, 0.5]
                }, {
                    "svr__kernel": ["rbf"],
                    "svr__C": [1, 10, 100],
                    "svr__epsilon": [0.1, 0.5],
                    "svr__gamma": ["scale", 0.1]
                }]
                risultatiSvr = nested_kfold_grid_src(pipe_svr, paramSvr, X, y, "SVR")
                print(f"{risultatiSvr['nome']} ha mae:{risultatiSvr['mae_mean']}, con std: {risultatiSvr['mae_std']}")
                print(f"mse: {risultatiSvr['mse_mean']}, con std: {risultatiSvr['mse_std']}")
                print(f"R^2: {risultatiSvr['r2_mean']}, con std: {risultatiSvr['r2_std']}")

                # --- KNN REGRESSOR con scaler + feature selection nella pipeline ---
                pipe_knn = Pipeline([
                    ('scaler', StandardScaler()),
                    ('feat_sel', SelectFromModel(
                        RandomForestRegressor(n_estimators=100, random_state=42),
                        threshold="median"
                    )),
                    ('model', KNeighborsRegressor())
                ])
                param_knn = {
                    'model__n_neighbors': [3, 5, 7, 11, 15],
                    'model__weights': ['uniform', 'distance'],
                    'model__p': [1, 2]
                }
                risultatiKnn = nested_kfold_grid_src(pipe_knn, param_knn, X, y, "KNN")
                print(f"{risultatiKnn['nome']} ha mae:{risultatiKnn['mae_mean']}, con std: {risultatiKnn['mae_std']}")
                print(f"mse: {risultatiKnn['mse_mean']}, con std: {risultatiKnn['mse_std']}")
                print(f"R^2: {risultatiKnn['r2_mean']}, con std: {risultatiKnn['r2_std']}")

                # --- ELASTIC NET con scaler + feature selection nella pipeline ---
                pipe_en = Pipeline([
                    ('scaler', StandardScaler()),
                    ('feat_sel', SelectFromModel(
                        RandomForestRegressor(n_estimators=100, random_state=42),
                        threshold="median"
                    )),
                    ('model', ElasticNet(max_iter=10000))
                ])
                param_en = {
                    'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'model__fit_intercept': [True, False]
                }
                risultatiEn = nested_kfold_grid_src(pipe_en, param_en, X, y, "ElasticNet")
                print(f"{risultatiEn['nome']} ha mae:{risultatiEn['mae_mean']}, con std: {risultatiEn['mae_std']}")
                print(f"mse: {risultatiEn['mse_mean']}, con std: {risultatiEn['mse_std']}")
                print(f"R^2: {risultatiEn['r2_mean']}, con std: {risultatiEn['r2_std']}")

                # --- BAYESIAN RIDGE con scaler + feature selection nella pipeline ---
                pipe_br = Pipeline([
                    ('scaler', StandardScaler()),
                    ('feat_sel', SelectFromModel(
                        RandomForestRegressor(n_estimators=100, random_state=42),
                        threshold="median"
                    )),
                    ('model', BayesianRidge())
                ])
                param_br = {
                    'model__max_iter': [300, 500],
                    'model__alpha_1': [1e-6, 1e-5, 1e-4],
                    'model__alpha_2': [1e-6, 1e-5, 1e-4],
                    'model__lambda_1': [1e-6, 1e-5, 1e-4],
                    'model__lambda_2': [1e-6, 1e-5, 1e-4]
                }
                risultatiBr = nested_kfold_grid_src(pipe_br, param_br, X, y, "BayesianRidge")
                print(f"{risultatiBr['nome']} ha mae:{risultatiBr['mae_mean']}, con std: {risultatiBr['mae_std']}")
                print(f"mse: {risultatiBr['mse_mean']}, con std: {risultatiBr['mse_std']}")
                print(f"R^2: {risultatiBr['r2_mean']}, con std: {risultatiBr['r2_std']}")

            case '8':
                df = load_dataset()
                df = cambia_target_evita_inf(df)
                df, _ = iqr_leggero(df, target="incidenti_per_km")
                X_train, X_valid, y_train, y_valid, X, y = dividiDataset(df, "incidenti_per_km")
                print(f'\nMedia del target "incidenti_per_km": {y.mean():.4f}')
                print(f'Mediana del target "incidenti_per_km": {y.median():.4f}')

                modelli = {
                    "GRADIENT BOOSTING": GradientBoostingRegressor(random_state=42),
                    "RANDOM FOREST": RandomForestRegressor(random_state=42),
                    "SVR + SCALER": Pipeline([('scaler', StandardScaler()), ('model', SVR(kernel='linear'))]),
                    "ElasticNet + SCALER": Pipeline(
                        [('scaler', StandardScaler()), ('model', ElasticNet(random_state=42))]),
                    "XGBoost": xgb.XGBRegressor(tree_method="hist", early_stopping_rounds=None),
                }

                for nome, modello in modelli.items():
                    print("\n" + "*" * 50)
                    print(f"FEATURE SELECTION con {nome}")
                    print("-" * 15)

                    # nuova firma: 8 valori restituiti
                    n_features_range, risultati_mae, risultati_mse, risultati_r2, \
                        miglior_num_features_mae, miglior_num_features_mse, \
                        miglior_num_features_r2, lista_feature = feat_selection(modello, X, y)

                    print(f"Miglior R²:  {max(risultati_r2):.4f} con {miglior_num_features_r2} feature")
                    print(f"Miglior MAE: {min(risultati_mae):.4f} con {miglior_num_features_mae} feature")
                    print(f"Feature scelte (R²): {lista_feature}")

                    ottieni_grafici_features_selection(
                        n_features_range, risultati_mae, risultati_mse, risultati_r2,
                        f"{nome} feat_sel nuovo target"
                    )
            case '10':

                ##QUESTA PARTE APPARTIENE ad un APPROCCIO non CONSIDERATO in questo STUDIO
                """
                ###*****************************************####
                #   APPROCCIO CON ONTOLOGIA  e TARGET CAMBIATO
                df = load_dataset()
                # df = cambiaTarget(df)
                df = cambia_target_evita_inf(df)
                df, _ = iqr_leggero(df, target="incidenti_per_km")
                # utilizzo di colonne dinamiche per la scelta degli scaglioni
                # si utilizzano featureche sono relative alla unghezza della strada, in quanto
                # sono gli unici che possono offrire un opinione se vengono presi singolarmente
                # con soglie generiche per tutti i segmenti
                colonne_strutturali = [
                    "poi_density",
                    "junction_density",
                    "urban_density",
                    "vulnerabilita",
                ]

                soglie_dati = calcola_soglie_dinamiche(df, colonne_strutturali)



                soglie_dati_puliti={}
                for col, vals in soglie_dati.items():
                    soglie_dati_puliti[col] = {
                        k: float(v) for k, v in vals.items()
                    }
                print(soglie_dati_puliti)

                condizioni_basso = (
                        (df['poi_density'] < soglie_dati['poi_density']['Q2']) &
                        (df['junction_density'] < soglie_dati['junction_density']['Q2']) &
                        (df['urban_density'] < soglie_dati['urban_density']['Q2']) &
                        (df['vulnerabilita'] < soglie_dati['vulnerabilita']['Q2'])
                )
                numero_basso_df = condizioni_basso.sum()
                print(
                    f"Segmenti che soddisfano la definizione di Basso nel DataFrame: {numero_basso_df} su {len(df)} totali")
                print(f"Percentuale: {numero_basso_df / len(df) * 100:.2f}%")
                soglia_basso = df['incidenti_per_km'].quantile(0.25)
                soglia_medio = df['incidenti_per_km'].quantile(0.50)
                soglia_alto = df['incidenti_per_km'].quantile(0.75)
                soglie_target = (soglia_basso, soglia_medio, soglia_alto)
                print("Soglie osservate:", soglie_target)
                onto = crea_ontologia_incidenti_con_validazione_esterna(df,soglie_dati_puliti)
                onto = popola_ontologia_nuovo_target(onto, df)
                with onto:
                    sync_reasoner(infer_property_values=True)

                verifica_sovrapposizioni(onto)
                risultati = evaluate_ontology_predictions(onto, soglie_target)
                print("Totale segmenti:", risultati['totale'])
                print("Confusion matrix:", risultati['matrice_confusione'])
                print("Errori gravi:", risultati['errori_gravi'])
                print("Tempo reasoning:", risultati['time_seconds'])
                print("")
                """

            case _:
                print("input sbagliato, ritenta")



def esperimenti():
    risposta = True

    while risposta:
        print("\n")
        print("1) torna indietro")
        print("2) esperimento con IQR sugli incidenti per km")
        inputUtente = input("\nScelta (1-2-3-4): ").strip()
        match inputUtente:
            case '1':
                print("Indietro")
                risposta = False

            case '2':
                df = load_dataset()
                #df = cambia_target_evita_inf(df)
                df["incidenti_per_km"] = np.where(
                    (df["lunghezza_m"] / 1000) != 0,
                    df["INCIDENTI"] / (df["lunghezza_m"] / 1000),
                    np.nan
                )
                df = df.dropna()
                df, _ = iqr_leggero(df, target="incidenti_per_km")
                #ora eliminiamo il targert incidenti_per_km
                df.drop(columns=["incidenti_per_km"], inplace=True)
                X_train, X_valid, y_train, y_valid, X, y = dividiDataset(df)
                bestModel = grid_search_SVR(X_train, y_train)
                risultatiGridSvr = k_fold_personalizzato(bestModel, X, y, "SVR con Grid Search New")
                ottieni_prestazioni_modelli([risultatiGridSvr])

            case _:
                print("input sbagliato, ritenta")