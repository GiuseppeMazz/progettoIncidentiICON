import pandas as pd
import os
from AnalizzaStrade import matrice_correlazione, arricchisciDataset
from Modelli import trainModelli, dividiDataset, grid_search_gradientBoosting, grid_search_randomForest, \
    grid_search_SVR, feat_selection, modello_finale
from ValutazioneModelli import ottieni_prestazioni_modelli, ottieni_grafici, k_fold_personalizzato, \
    ottieni_grafici_features_selection
from PreparaDataset import file_esterno, process_incidents_by_province
from Predizione import df_predizione_utente, carica_modello, effettua_predizione
from Modelli import salva_modello, prestazioni_modelli_gridSearch

# questo file permette si utilizare tutte le funzioni interagendo con l'utente tramite menu a scelta



def load_dataset():
    pathDataset = "dataset_finale.csv"
    df = pd.read_csv(pathDataset, sep=";", decimal=",", encoding="utf-8-sig")
    return df

def lavora_con_modelli():
    risposta = True
    while risposta:
        print("\n")
        print("1) torna indietro")
        print("2) training e risultati dei modelli base (richiede un po' di tempo)")
        print("3) crea grafici (nella cartella 'grafici') dei modelli di ML base")
        print("4) crea grafici (nella cartella 'grafici') dei modelli con GridSearch ATTENZIONE: richiede tempo")
        print("5) modelli grid search")
        print("6) effettua predizione")
        print("7) salva modello per predizioni (inutile se già presente, come dovrebbe)")
        print("8) produci grafici di feature selection progressiva")


        inputUtente = input("\nScelta (1-2-3-4-5-6-7-8): ").strip()
        match inputUtente:

            case '1':
                print("Indietro")
                risposta = False

            case '2':

                df = load_dataset()
                X_train, X_valid, y_train, y_valid, X, y = dividiDataset(df)
                risultati = trainModelli(X,y)
                ottieni_prestazioni_modelli(risultati)

            case '3':
                df = load_dataset()
                X_train, X_valid, y_train, y_valid, X, y = dividiDataset(df)
                risultati = trainModelli(X, y)
                ottieni_grafici(risultati)

            case '4':
                df =load_dataset()
                X_train, X_valid, y_train, y_valid, X, y = dividiDataset(df)
                risultati= prestazioni_modelli_gridSearch(X_train, X_valid, y_train, y_valid, X, y)
                ottieni_grafici(risultati)

            case '5':
                risposta_gs=True

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
                            X_train, X_valid, y_train, y_valid, X, y = dividiDataset(df)
                            bestModel = grid_search_gradientBoosting(X_train, y_train)
                            risultatiGridGb=k_fold_personalizzato(bestModel, X, y, "Gradient Boosting con Grid Search")
                            ottieni_prestazioni_modelli([risultatiGridGb])

                        case'3':
                            df = load_dataset()
                            X_train, X_valid, y_train, y_valid, X, y = dividiDataset(df)
                            bestModel = grid_search_randomForest(X_train, y_train)
                            risultatiGridRf = k_fold_personalizzato(bestModel, X, y, "Random Forest con Grid Search")
                            ottieni_prestazioni_modelli([risultatiGridRf])
                        case '4':
                            df = load_dataset()
                            X_train, X_valid, y_train, y_valid, X, y = dividiDataset(df)
                            bestModel = grid_search_SVR(X_train, y_train)
                            risultatiGridSvr = k_fold_personalizzato(bestModel, X, y, "SVR con Grid Search")
                            ottieni_prestazioni_modelli([risultatiGridSvr])


                        case _:
                            print("input sbagliato, ritenta")

            case '6':
                risposta_pred = True

                while risposta_pred:

                    print("\n")
                    print("1) indietro")
                    print("2) dai in input Milano - Bergamo  (45.4642, 9.1910), (45.7040, 9.6630) [richiede 3 min. circa]")
                    print("3) dai in input Bari - Bisceglie (41.1108, 16.8741), (41.2373, 16.5030) [richiede 2 min. circa]" )
                    print("4) dai in input Palermo - Messina  (38.0912, 13.3623), (38.1895, 15.5469) [richiede 30 min. circa]")
                    print("5) dai in input Arezzo - Viterbo (43.4582, 11.8899), (42.4507, 12.0890) [richiede 7 min. circa]")
                    print("6) dai in input Savona - Genova (44.3189, 8.4845), (44.4197, 8.9061) [richiede 2 min. circa]")

                    print("\nATTENZIONE: ovviamente punti molto distanti possono chiedere abbastanza tempo")
                    inputUtente = input("\nScelta (1-2-3-4-5-6): ").strip()

                    match inputUtente:
                        case '1':
                            print("Indietro")
                            risposta_pred = False
                        case '2':

                            effettua_predizione("Milano - Bergamo", (45.4642, 9.1910), (45.7040, 9.6630), load_dataset())
                        case '3':
                            effettua_predizione("Bari - Bisceglie", (41.1108, 16.8741), (41.237393, 16.5030), load_dataset())

                        case '4':
                            effettua_predizione("Palermo - Messina", (38.0912, 13.3623), (38.1895, 15.5469), load_dataset())

                        case '5':
                            effettua_predizione("Arezzo - Viterbo", (43.4582, 11.8899), (42.4507, 12.0890), load_dataset())

                        case'6':
                            effettua_predizione("Savona - Genova",  (44.3189, 8.4845), (44.4197, 8.9061), load_dataset())

                        case _:
                            print("input sbagliato, ritenta")

            case '7':
                df = load_dataset()#bestk =26
                X_train, X_valid, y_train, y_valid, X, y = dividiDataset(df)
                pipeline_finale = modello_finale(grid_search_SVR(X_train, y_train))
                #pipeline.fit(X_train, y_train)
                print(f"Parametri selezionati{pipeline_finale.get_params()}")
                risultati_modFin_SVR = k_fold_personalizzato(pipeline_finale, X, y, "SVR con Grid Search FeatureSel")
                ottieni_prestazioni_modelli([risultati_modFin_SVR])

                pipeline_finale.fit(X,y)
                #ottenere le feature scelte
                selector = pipeline_finale.named_steps["kbest"]
                mask = selector.get_support()
                selected_features = X.columns[mask]
                print("Feature sclete: ")
                print(selected_features)


                salva_modello(pipeline_finale,"modelloSVR_GridSrch_FeatSel.joblib")


            case '8':
                menu_features_selection()
            case _:
                print("input sbagliato, ritenta")


def menu_features_selection():
    risposta_feat_sel =True
    while risposta_feat_sel:
        print("\n")
        print("1) torna indietro")
        print("2) ottieni grafico features selection Random Forest")
        print("3) ottieni grafico features selection Grad Boosting ")
        print("4) ottieni grafico features selection SVR  ")


        inputUtenteFeatSel = input("\nScelta (1-2-3-4): ").strip()
        match inputUtenteFeatSel:

            case '1':
                print("Indietro")
                risposta_feat_sel = False


            case '2':
                df = load_dataset()
                X_train, X_valid, y_train, y_valid, X, y = dividiDataset(df)
                bestModel = grid_search_randomForest(X_train, y_train)

                print("Gridsearch finita")
                n_features_range, risultati_mae, risultati_mse,miglior_num_features_mae, miglior_num_features_mse\
                    = feat_selection(bestModel, X, y)

                print(f"Miglior numero di feature minimizzando mae: {miglior_num_features_mae}")
                print(f"Miglior numero di feature minimizzando mse: {miglior_num_features_mse}")
                print(f"Miglior mae media: {min(risultati_mae)}")
                print(f"Miglior mse media: {min(risultati_mse)}")
                ottieni_grafici_features_selection(n_features_range, risultati_mae, risultati_mse, "RandForest feature selection")

            case '3':
                df = load_dataset()
                X_train, X_valid, y_train, y_valid, X, y = dividiDataset(df)
                bestModel = grid_search_gradientBoosting(X_train, y_train)
                n_features_range, risultati_mae, risultati_mse, miglior_num_features_mae, miglior_num_features_mse \
                    = feat_selection(bestModel, X, y)
                print(f"Miglior numero di feature minimizzando mae: {miglior_num_features_mae}")
                print(f"Miglior numero di feature minimizzando mse: {miglior_num_features_mse}")
                print(f"Miglior mae media: {min(risultati_mae)}")
                print(f"Miglior mse media: {min(risultati_mse)}")
                ottieni_grafici_features_selection(n_features_range, risultati_mae, risultati_mse,
                                                   "GradBoosting feature selection")

            case '4':
                df = load_dataset()
                X_train, X_valid, y_train, y_valid, X, y = dividiDataset(df)
                bestModel = grid_search_SVR(X_train, y_train)
                n_features_range, risultati_mae, risultati_mse, miglior_num_features_mae, miglior_num_features_mse \
                    = feat_selection(bestModel, X, y)
                print(f"Miglior numero di feature minimizzando mae: {miglior_num_features_mae}")
                print(f"Miglior numero di feature minimizzando mse: {miglior_num_features_mse}")
                print(f"Miglior mae media: {min(risultati_mae)}")
                print(f"Miglior mse media: {min(risultati_mse)}")
                ottieni_grafici_features_selection(n_features_range, risultati_mae, risultati_mse,
                                                   "SVR feature selection")

            case _:
                print("input sbagliato, ritenta")


def esplora_vecchio_dataset():
    risposta = True
    while risposta:
        print("\n")
        print('1) indietro')
        print('2) esplora dataset iniziale')
        print("3) produci dataset partendo dall'originale ATTENZIONE: può richiedere davvero tante ore")
        print('4) ')

        inputUtente = input("\nScelta (1-2-3-4): ").strip()
        match inputUtente:

            case '1':
                print("indietro")
                risposta = False

            case '2':
                print('\n\n')
                df = file_esterno()
                pd.set_option('display.max_columns', None)
                print(df.describe().T)
            case '3':
                df = file_esterno()
                df_raggruppato = process_incidents_by_province(df)
                dataset_finale = arricchisciDataset(df_raggruppato)
                os.makedirs("dataset_costruito ", exist_ok=True)
                dataset_finale.to_csv(f"dataset_costruito/dataset_finale_costruito.csv ",
                                      index=False, sep=';', decimal=',', encoding='utf-8-sig')
            case '4':
                print("")

            case _:
                print("input sbagliato, ritenta")





if __name__ == "__main__":
    print("MODELLI per la PREDIZIONE DELLA PERICOLOSITA' di una STRADA")
    print("\n-Navigare per il menù tramite input in console ")
    print('inserire il numero corrispettivo alla opzione desiderata ')


    risposta = True
    while risposta:
        print("\n")
        print('1) chiusura programma')
        print('2) ottieni descrizione dataset (finale)')
        print('3) dataset originale')
        print('4) lavora con i modelli')
        print('5) produci matrice di correlazione')


        inputUtente = input("\nScelta (1-2-3-4-5): ").strip()
        match inputUtente:
            case '1':
                print("chiusura")
                risposta = False

            case '2':
                print('\n\n')
                df = load_dataset()
                pd.set_option('display.max_columns', None)
                print(df.describe().T)

            case '3':
                esplora_vecchio_dataset()
            case '4':
                lavora_con_modelli()
            case '5':
                df = load_dataset()
                matrice = matrice_correlazione(df)
                os.makedirs("grafici", exist_ok=True)
                matrice.savefig(f"grafici/matriceDiCorrelazionePearson.png", dpi=300, bbox_inches="tight")
            case _:
                print("input sbagliato, ritenta")
