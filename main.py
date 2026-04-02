import pandas as pd
from AnalizzaStrade import matrice_correlazione, arricchisciDataset, iqr_leggero
from CambioDataset import lavora_con_target_nuovo, cambia_target_evita_inf
from Modelli import trainModelli, dividiDataset, feat_selection,\
    gridSrc_con_featSel_integrata, salva_modello_finale_GB
from ValutazioneModelli import ottieni_prestazioni_modelli, ottieni_grafici, \
    ottieni_grafici_features_selection
from PreparaDataset import file_esterno, process_incidents_by_province, load_dataset
from Predizione import effettua_predizione
from Modelli import prestazioni_modelli_gridSearch
from classificazione.Classi import lavora_con_classificazione
#from resoning.ReasoningStrade import usa_onto, crea_ontologia_segmenti
#from resoning.Swrl import analizza_completo, crea_ontologia_incidenti, popola_ontologia, classifica_rischio, \
#     test_ontologia, crea_ontologia_incidenti_con_validazione, \
#    popola_ontologia_con_validazione, valuta_accuratezza, crea_ontologia_incidenti_con_validazione_esterna, \
#    evaluate_ontology_predictions

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from owlready2 import *


# Questo file permette di utilizzare tutte le funzioni interagendo con l'utente tramite menu a scelta


def lavora_con_modelli():
    risposta = True

    while risposta:
        print("\n")
        print("1) torna indietro")
        print("2) training e risultati dei modelli base (richiede un po' di tempo)")
        print("3) crea grafici (nella cartella 'grafici') dei modelli di ML base")
        print("4) crea grafici (nella cartella 'grafici') dei modelli con GridSearch ATTENZIONE: richiede tempo")
        print("5) feature selection")
        print("6) modelli feat selection -> grid search")
        print("7) effettua predizione")
        print("8) salva modello per predizioni (inutile se già presente, come dovrebbe)")



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
                # feature selection sui modelli base, stampa a schermo
                df = load_dataset()
                X_train, X_valid, y_train, y_valid, X, y = dividiDataset(df)

                modelli = {
                    "Random Forest": RandomForestRegressor(random_state=42),
                    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                    "SVR": Pipeline([('scaler', StandardScaler()),('model', SVR(kernel='linear'))])
                }

                for nome, modello in modelli.items():
                    print(f"\n--- Feature selection con {nome} ---")
                    n_features_range, risultati_mae, risultati_mse, risultati_r2, \
                        miglior_num_features_mae, miglior_num_features_mse, miglior_num_features_r2, \
                        lista_feature = feat_selection(modello, X, y)

                    #print(f"  Miglior R2: {max(risultati_r2):.4f} con {miglior_num_features_r2} feature")
                    #print(f"  Feature selezionate: {list(lista_feature)}")  # copia da qui

                    ottieni_grafici_features_selection(
                        n_features_range, risultati_mae, risultati_mse, risultati_r2, f"{nome} feature selection"
                    )

            case '6':
                df = load_dataset()
                gridSrc_con_featSel_integrata(df)

            case '7':
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

            case '8':
                df = load_dataset()
                salva_modello_finale_GB(df)


            case _:
                print("input sbagliato, ritenta")


#############
#   Questa funzione appartiene a test NON è previsto il suo utilizzo se questo messaggio è ancora PRESENTE
#####
"""
def lavora_con_ontologia():
    risposta = True
    while risposta:
        print("\n")
        print('1) indietro')
        print("2) effettua predizioni")
        print("3) mini analisi dataset")
        print("4) esperimento swrl")
        print("5)")
        print("6) utilizzo di ontologia complessa con controllo qualità predizioni")
        print("7) utilizzo di ontologia con validazione esterna")

        inputUtente = input("\nScelta (1-2-3-4): ").strip()
        match inputUtente:

            case '1':
                print("Indietro")
                risposta = False

            case '2':
                df = load_dataset()
                df_finale = df.copy()  # dataset dopo feature engineering

                onto = crea_ontologia_segmenti(df_finale)
                usa_onto(onto)

            case '3':
                df = load_dataset()
                df_finale = df
                print(df_finale["limiteVel"].describe())
                print(df_finale["complexity_index"].describe())
                print(df_finale["poi_totali"].describe())
                print(df_finale["giunzione"].describe())

                analizza_completo(df)
            case '4':
                df = load_dataset()
                df_finale = df.copy()  # dataset dopo feature engineering

                print("Avvio ontologia semplificata...")

                #
                #onto = crea_ontologia_con_reasoner(df)
                # Salva risultato
                #onto.save("incidenti_classificati.owl")
                #print("\n Ontologia salvata in 'incidenti_classificati.owl'")


                ##ONTOLOGIA MOLTO COMPLICATA
                # 1. Crea l'ontologia
                onto = crea_ontologia_incidenti(df)
                onto = popola_ontologia(onto, df)
                onto = classifica_rischio(onto)
            case '5':
                #onto = visualizza_ontologia_python("incidenti_classificati.owl")
                #plot_distribuzione_classi(onto)
                #plot_torta_classi(onto)
                #plot_venn_classi(onto)

                onto = get_ontology("incidenti_stradali.owl").load()
                df = load_dataset()
                df_finale = df.copy()
                test_ontologia(onto, df_finale)

            case '6':
                df = load_dataset()
                df_finale = df.copy()
                print("\n FASE 2: Creazione ontologia con validazione...")

                onto_val = crea_ontologia_incidenti_con_validazione(df_finale)
                print("\n FASE 3: Popolamento ontologia con dati...")
                onto_val = popola_ontologia_con_validazione(onto_val, df_finale)
                #valutazione accuratezza
                risultati_validazione = valuta_accuratezza(onto_val)

                if risultati_validazione:
                    print("\n FASE 5: Salvataggio risultati...")

                    # Salva ontologia
                    onto_val.save("ontologia_incidenti_validata.owl")
                    print(" Ontologia salvata in 'ontologia_incidenti_validata.owl'")

            case "7":
                df = load_dataset()
                df_finale = df.copy()

                soglia_basso = float(df_finale['INCIDENTI'].quantile(0.25))
                soglia_medio = float(df_finale['INCIDENTI'].quantile(0.50))
                soglia_alto = float(df_finale['INCIDENTI'].quantile(0.75))
                soglie = (soglia_basso, soglia_medio, soglia_alto)

                print("Soglie osservate:", soglie)
                onto = crea_ontologia_incidenti_con_validazione_esterna(df_finale)
                #onto = popola_ontologia_con_val_esterna(onto, df_finale)

                with onto:
                    sync_reasoner(infer_property_values=True)

                risultati = evaluate_ontology_predictions(onto, soglie)

                # 6. Stampa risultati
                print("Totale segmenti:", risultati['totale'])
                print("Confusion matrix:", risultati['matrice_confusione'])
                print("Errori gravi:", risultati['errori_gravi'])
                print("Tempo reasoning:", risultati['time_seconds'])

            case _:
                    print("input sbagliato, ritenta")
                    """
##### fine della parte di codice appartenente ai test
###########

def esplora_vecchio_dataset():
    risposta = True
    while risposta:
        print("\n")
        print('1) indietro')
        print('2) esplora dataset iniziale')
        print("3) produci dataset partendo dall'originale ATTENZIONE: può richiedere davvero tante ore")

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
        print('4) lavora con i modelli e target incidenti')
        print('5) produci matrice di correlazione')
        print('6) approccio con taraget incidenti/km')
        print('7) approccio con classificazione')
        #print('8) (PROVE) lavora con ontologia')

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
            case '6':
                lavora_con_target_nuovo()
            case '7':
                df = load_dataset()
                # df = cambiaTarget(df)
                df = cambia_target_evita_inf(df)
                df, _ = iqr_leggero(df, target="incidenti_per_km")
                lavora_con_classificazione(df)

            case '8':
                #esperimenti con ONTOLOGIA
                print("Questo approccio ad ora è stato eliminato dal progetto")
                #lavora_con_ontologia()

            case _:
                print("input sbagliato, ritenta")
