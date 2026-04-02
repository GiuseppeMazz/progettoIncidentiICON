from dataclasses import dataclass

import pandas as pd

#file principalmente usato per la pulizia iniziale del vecchio datasetdataset originale


def load_dataset():
    pathDataset = "dataset_finale.csv"
    df = pd.read_csv(pathDataset, sep=";", decimal=",", encoding="utf-8-sig")
    return df



def file_esterno():
    path_incidenti = "Localizzazione_incidenti_stradali-2023_2.CSV"
    incidenti_data = pd.read_csv(path_incidenti, sep=";", encoding="latin1")
    #tolti gli incidenti appartenenti ad anni diversi dal 2023 (per motivi di modifiche applicate al file excel)
    incidenti_data.drop(incidenti_data[incidenti_data["INCIDENTI"] == 0].index, inplace=True)
    # selezione di alcune regioni
    incidenti_regioni = incidenti_data[incidenti_data["REGIONE"].isin(["Puglia", "Trentino-Alto Adige",
                                                                           "Abruzzo", "Umbria", "Piemonte", "Toscana",
                                                                           "Lazio", "Campania", "Calabria", "Veneto",
                                                                           "Liguria", "Marche", "Molise", "Sicilia"])]

    return incidenti_regioni



@dataclass
class TuplaIncidenti:
    COD_REG: 'int'
    REGIONE: 'str'
    COD_PROV: 'int'
    PROVINCIA: 'str'
    CODICE: 'int'
    NOME_STRADA: 'str'
    DA_KM: 'float'
    A_KM: 'float'
    INCIDENTI: 'int'
    INCIDENTI_MORTALI: 'int'
    MORTI: 'int'
    FERITI: 'int'
    NVEI: 'int'  # numero veicoli(?)
    Inc2RMoto: 'int'
    IncMerci: 'int'
    IncBici_e_Micromobilità: 'int'
    ScFRONTALE: 'int'
    SCFRONTLAT_ScLAT: 'int'
    TAMPONAMENTO: 'int'
    InvPedoni: 'int'
    FUORIUSCITA: 'int'
    ALTRO: 'int'

    def __init__(self,array):
        self.COD_REG = array["COD REG"]
        self.REGIONE= array["REGIONE"]
        self.COD_PROV= array["COD PROV"]
        self.PROVINCIA= array["PROVINCIA"]
        self.CODICE= array["CODICE"]
        self.NOME_STRADA = array["NOME STRADA"]
        self.DA_KM = array["DA KM"]
        self.A_KM = array["A KM"]
        self.INCIDENTI =  array["INCIDENTI"]
        self.INCIDENTI_MORTALI = array["INCIDENTI MORTALI"]
        self.MORTI = array["MORTI"]
        self.FERITI = array["FERITI"]
        self.NVEI = array["NVEI"]#numero veicoli(?)
        self.Inc2RMoto = array["Inc2RMoto"]
        self.IncMerci = array["IncMerci"]
        self.IncBici_e_Micromobilità = array["IncBici e Micromobilità"]
        self.ScFRONTALE = array["ScFRONTALE"]
        self.SCFRONTLAT_ScLAT = array["ScFRONTLAT-ScLAT"]
        self.TAMPONAMENTO = array["TAMPONAMENTO"]
        self.InvPedoni = array["InvPEDONI"]
        self.FUORIUSCITA = array["FUORIUSCITA"]
        self.ALTRO = array["ALTRO"]

    def aggiornaIncidenti(self, array):
        self.INCIDENTI += array["INCIDENTI"]
        self.INCIDENTI_MORTALI += array["INCIDENTI MORTALI"]
        self.MORTI += array["MORTI"]
        self.FERITI += array["FERITI"]
        self.NVEI += array["NVEI"]  # numero veicoli(?)
        self.Inc2RMoto += array["Inc2RMoto"]
        self.IncMerci += array["IncMerci"]
        self.IncBici_e_Micromobilità += array["IncBici e Micromobilità"]
        self.ScFRONTALE += array["ScFRONTALE"]
        self.SCFRONTLAT_ScLAT += array["ScFRONTLAT-ScLAT"]
        self.TAMPONAMENTO += array["TAMPONAMENTO"]
        self.InvPedoni += array["InvPEDONI"]
        self.FUORIUSCITA += array["FUORIUSCITA"]
        self.ALTRO += array["ALTRO"]
        self.A_KM = array["A KM"]

    def setKM(self, array):
        self.A_KM = array["A KM"]

    def getTupla(self):
        array=[ self.COD_REG, self.REGIONE, self.COD_PROV, self.PROVINCIA, self.CODICE, self.NOME_STRADA,
               self.DA_KM, self.A_KM, self.INCIDENTI, self.INCIDENTI_MORTALI,self.MORTI, self.FERITI, self.NVEI ,#numero veicoli(?)
               self.Inc2RMoto, self.IncMerci, self.IncBici_e_Micromobilità, self.ScFRONTALE, self.SCFRONTLAT_ScLAT,
               self.TAMPONAMENTO, self.InvPedoni, self.FUORIUSCITA, self.ALTRO]
        return array



def process_incidents_by_province(incidenti_df):
    # Raggruppa incidenti per comune
    province = incidenti_df['PROVINCIA'].unique()
    df_nuovo = pd.DataFrame(columns= incidenti_df.columns)


    #ciclo sulle province
    for provincia in province:
        print(f"Processando {provincia}")
        cod_strada_attuale=""
        strada = None

        #ciclo sulle righe degli incidenti
        for index, row in incidenti_df.iterrows():

            if row['PROVINCIA'] == str(provincia):
                if cod_strada_attuale != row['CODICE']:
                    if strada is not None:
                        df_nuovo.loc[len(df_nuovo)] = strada.getTupla()
                    #print(f"Processando strada: " + row["CODICE"])
                    strada = TuplaIncidenti(row)
                    cod_strada_attuale = row['CODICE']

                elif  cod_strada_attuale == row['CODICE']:
                    #print(f"Processando strada: " + row["CODICE"])
                    strada.aggiornaIncidenti(row)
        #per l'ulitma strada(non trova una dopo con codice diverso)(non è detto che funzioni in ogni caso?? si?)
        df_nuovo.loc[len(df_nuovo)] = strada.getTupla()
    return df_nuovo

def raggruppa_incidenti(df):
    return process_incidents_by_province(df)

def aggiungi_colonne(df):
    df[['limiteVel','numeroCorsie','sensoUnico','ponte','giunzione','highway_accessi','tunnel','width',
        'highway_motorway','highway_trunk','hyghway_primary','highway_scondary','highway_residential']] = None
    return df
