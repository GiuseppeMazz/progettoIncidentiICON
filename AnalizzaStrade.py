import numpy as np
import pandas as pd
import osmnx as ox
from shapely.ops import linemerge, unary_union
import geopandas as gpd
import re
from shapely.geometry import LineString, MultiLineString
import time
import seaborn as sns
import matplotlib.pyplot as plt

#file principalmente usato per prelevare le informazioni dell'ontologia
#tramite query usando OSMNX e arricchire il dataset

def arricchisciDataset(dataset_grezzo):
    dataset= aggiungi_colonne_strade(dataset_grezzo)
    pd.set_option('display.max_columns', None)
    print(dataset_grezzo.describe().T)
    dataset = analisi_completa_dataset(dataset)
    dataset = rendi_numeric(dataset)
    dataset = pulizia_info_pregresse(dataset)
    dataset = add_engineered_features(dataset)
    dataset, df_outliers = iqr_leggero(dataset)
    return dataset

def is_all_na_or_empty(x):
    if x is None:
        return True
    if not isinstance(x, (list, tuple, set, np.ndarray, pd.Series)):
        return pd.isna(x)
    arr = np.asarray(x)
    if arr.size == 0:
        return True
    return np.all(pd.isna(arr))

def puliziaNomeStrada(codice):
    codicePulito = str(codice).split("-")[0].replace(" ","")
    if re.match("^SS0", codicePulito):
        codicePulito = codicePulito[:2] + codicePulito[3:]
    elif re.match("^A0", codicePulito):
        codicePulito = codicePulito[:1] + codicePulito[2:]
    return codicePulito


def inizializza_colonne_strutture(df):
    categorie = ['scuole','ospedali','polizia','pompieri','bus','supermercati','farmacie','parcheggi',
                 'ristoranti','banche']
    for categoria in categorie:
        col=f"strutture_{categoria}"
        if col not in df.columns:
            df[col] = 0

    return df


def aggiungi_colonne_strade(df):
    df[["limiteVel", "numeroCorsie", "sensoUnico", "ponte", "giunzione", "highway_accessi", "tunnel",
                   "width", "highway_motorway", "highway_trunk", "highway_primary",
                   "highway_secondary", "highway_residential"]] = None
    return df


def safe_merge(geometry):

    if geometry is None:
        return None

    # accetta sia una singola geometria che una lista/GeoSeries
    if isinstance(geometry, (LineString, MultiLineString)):
        geom_list = [geometry]
    else:
        geom_list = list(geometry)

    geom_list = [g for g in geom_list if g is not None and not g.is_empty]
    if not geom_list:
        return None

    try:
        merged = linemerge(geom_list)
    except Exception:
        merged = unary_union(geom_list)

    if isinstance(merged, LineString):
        return merged if not merged.is_empty else None

    #se la lista è frammentata prendi tutto se fatta da pezzi piccoli
    # altrimenti prendi quelli di 1km di lunghezza
    if isinstance(merged, MultiLineString):
        parts = [g for g in merged.geoms if not g.is_empty]

        long_parts = [g for g in parts if g.length >= 1000]

        if long_parts:
            return unary_union(long_parts)

        return unary_union(parts)

    return None


def getInfoStrada(cod_strada, vertici):
    try:
        mask = (
            vertici['ref'].fillna('').str.contains(cod_strada, case=False, regex=True) |
            vertici['name'].fillna('').str.contains(cod_strada, case=False, regex=True)
        )

        vertici_strada = vertici[mask]

        if vertici_strada.empty:
            return None

        vertici_metric = vertici_strada.to_crs(epsg=32633)
        vertici_metric['length_m'] = vertici_metric.geometry.length
        lunghezza_totale = vertici_metric['length_m'].sum()

        try:
            linea_strada = linemerge(list(vertici_strada.geometry))
        except:
            linea_strada = unary_union(vertici_strada.geometry)

        gdf = gpd.GeoDataFrame(geometry = [linea_strada], crs ='EPSG:4326').to_crs('EPSG:32633')

        return gdf, lunghezza_totale
    except Exception as e:
        print(f'Errore in getInfoStrada per {cod_strada}: {e}')
        return None



def trova_strutture_vicine_completa(
    merged_geom,
    crs_strada,
    distanza_max=400,
    passo_km=5,
    nome_strada=None
):

    categorie_map = {
        'scuole':       {'amenity': 'school'},
        'ospedali':     {'amenity': 'hospital'},
        'polizia':      {'amenity': 'police'},
        'pompieri':     {'amenity': 'fire_station'},
        'bus':          {'highway': 'bus_stop'},
        'supermercati': {'shop': 'supermarket'},
        'farmacie':     {'amenity': 'pharmacy'},
        'parcheggi':    {'amenity': 'parking'},
        'ristoranti':   {'amenity': 'restaurant'},
        'banche':       {'amenity': 'bank'},
    }

    fallback = {k: 0 for k in categorie_map}

    try:
        if merged_geom is None or merged_geom.is_empty:
            print(f"  - Geometria non valida per {nome_strada}")
            return fallback

        # Normalizza geometria
        geom = safe_merge(merged_geom)
        if geom is None:
            print(f"  - Geometria non normalizzabile per {nome_strada}")
            return fallback

        # GeoDataFrame in CRS strada
        gdf = gpd.GeoDataFrame(geometry=[geom], crs=crs_strada)

        # CRS metrico FISSO (Italia)
        gdf_m = gdf.to_crs(epsg=32633)
        geom_m = gdf_m.geometry.iloc[0]

        if not np.isfinite(geom_m.length) or geom_m.length <= 0:
            print(f"  - Lunghezza non valida per {nome_strada}")
            return fallback

        # BUFFER della LINEA
        buffer_m = geom_m.buffer(distanza_max)

        # Torna in WGS84 per OSM
        buffer_wgs = gpd.GeoSeries([buffer_m], crs=32633).to_crs(epsg=4326).iloc[0]

        print(
            f"  - Analisi {nome_strada or 'strada'} "
            f"({geom_m.length/1000:.1f} km, buffer {distanza_max} m)"
        )

        # QUERY UNICA OSM
        tags_unici = {
            'amenity': [
                'school', 'hospital', 'police', 'fire_station',
                'pharmacy', 'parking', 'restaurant', 'bank'
            ],
            'shop': ['supermarket'],
            'highway': ['bus_stop']
        }

        pois = ox.features_from_polygon(buffer_wgs, tags=tags_unici)

        if pois.empty:
            return fallback

        # CONTEGGIO
        conteggi = fallback.copy()

        if 'amenity' in pois.columns:
            conteggi['scuole']     = (pois['amenity'] == 'school').sum()
            conteggi['ospedali']   = (pois['amenity'] == 'hospital').sum()
            conteggi['polizia']    = (pois['amenity'] == 'police').sum()
            conteggi['pompieri']   = (pois['amenity'] == 'fire_station').sum()
            conteggi['farmacie']   = (pois['amenity'] == 'pharmacy').sum()
            conteggi['parcheggi']  = (pois['amenity'] == 'parking').sum()
            conteggi['ristoranti'] = (pois['amenity'] == 'restaurant').sum()
            conteggi['banche']     = (pois['amenity'] == 'bank').sum()

        if 'shop' in pois.columns:
            conteggi['supermercati'] = (pois['shop'] == 'supermarket').sum()

        if 'highway' in pois.columns:
            conteggi['bus'] = (pois['highway'] == 'bus_stop').sum()

        return conteggi

    except Exception as e:
        print(f"  - Errore strutture vicine {nome_strada}: {e}")
        return fallback


def analizza_provincia_completa(provincia_nome, df_strade_provincia, provincia_geometria = None):
    print(f'\n--- Analisi provincia: {provincia_nome} ---')

    try:
        #Scarica dati OSM per la provincia
        print(f'Scarico dati OSM per {provincia_nome}...')

        if provincia_geometria is not None:
            poly = provincia_geometria
        else:
            try:
                area = ox.geocode_to_gpd(f'{provincia_nome}, Italia')
                poly = area.geometry.iloc[0]
            except:
                graph = ox.graph_from_place(f'{provincia_nome}, Italia', network_type = 'drive',simplify = True)
                vertici = ox.graph_to_gdfs(graph, nodes=False, edges=True)
                print(f'Trovati {len(vertici)} segmenti stradali')
                return vertici

        # Scarica rete stradale
        graph = ox.graph_from_polygon(poly, network_type='drive', simplify=True, retain_all=True)
        vertici = ox.graph_to_gdfs(graph, nodes=False, edges=True)
        print(f'Trovati{len(vertici)} segmenti stradali')

    except Exception as e:
        print(f'Errore scaricamento dati OSM: {e}')
        return None

    risultati = []

    for idx, row in df_strade_provincia.iterrows():
        try:
            codice_pulito = puliziaNomeStrada(row['NOME STRADA'])
            codice_originale = row['CODICE']

            print(f"\n{codice_originale} ({codice_pulito}):")

            info_strada = getInfoStrada(codice_pulito, vertici)

            if info_strada is None:
                print(f'  Non trovata in OSM')

                risultati.append({
                    'indice_originale': idx,
                    'codice': codice_originale,
                    'nome_pulito': codice_pulito,
                    'lunghezza_m':0,
                    'trovata': False
                })
                continue

            gdf_strada, lunghezza = info_strada
            print(f'  Trovata: {lunghezza:.0f}m ({lunghezza/1000:.1f}km)')


            geom = gdf_strada.geometry.iloc[0]
            merged = safe_merge(geom)


            conteggi_strutture = trova_strutture_vicine_completa(
                merged,
                gdf_strada.crs,
                distanza_max=400,
                passo_km=5,
                nome_strada=codice_pulito
            )

            risultato = {
                'indice_originale': idx,
                'codice': codice_originale,
                'nome_pulito': codice_pulito,
                'lunghezza_m': lunghezza,
                'trovata': True
            }

            # Conteggi strutture
            for cat, count in conteggi_strutture.items():
                risultato[f'strutture_{cat}'] = count

            risultati.append(risultato)

            # Pausa per evitare il "rate limiting" cercare di limitare
            time.sleep(1)

        except Exception as e:
            print(f'  Errore: {e}')
            continue

    return vertici, risultati


def analisi_completa_dataset(df):

    print('='*10)
    print('ANALISI COMPLETA DATASET STRADE')
    print('='*10)

    # Inizializza colonne
    df = inizializza_colonne_strutture(df)

    colonne_info = ['limiteVel', 'numeroCorsie', 'sensoUnico', 'ponte',
                    'giunzione', 'tunnel', 'width', 'lunghezza_m']

    for col in colonne_info:
        if col not in df.columns:
            df[col] = None


    highway_categories = ['motorway', 'trunk', 'primary', 'secondary',
                          'residential', 'accessi']

    for cat in highway_categories:
        col = f'highway_{cat}'
        if col not in df.columns:
            df[col] = None

    # Ottieni regioni uniche
    regioni = df['REGIONE'].unique()
    print(f'Regioni trovate: {len(regioni)}')
    print(f'Strade totali: {len(df)}')

    indici_da_eliminare = []
    risultati_completi = []

    totale_strade = 0
    strade_trovate = 0
    strade_con_strutture = 0

    for regione in regioni:
        print(f"\n{'='*80}")
        print(f'REGIONE: {regione}')
        print(f"{'='*80}")

        df_regione = df[df['REGIONE'] == regione].copy()
        print(f'Strade in regione: {len(df_regione)}')

        try:
            print(f'Ottengo province per {regione}...')
            province_regione = ox.features_from_place(f'{regione}, Italia', tags={'admin_level': '6'})
            #print(f'Trovate {len(province_regione)} ')
        except Exception as e:
            print(f'Errore ottenendo province: {e}')
            continue

        province_regione_lista = df_regione['PROVINCIA'].unique()

        for provincia in province_regione_lista:
            print(f'\n{'='*60}')
            print(f'PROVINCIA: {provincia}')
            print(f'{'='*60}')

            # Trova geometria provincia
            provincia_trovata = False
            poly_provincia = None

            if 'name' in province_regione.columns:
                mask = province_regione['name'].str.contains(provincia, case=False, na=False)
                if mask.any():
                    prov = province_regione[mask]
                    if not prov.empty:
                        poly_provincia = prov.geometry.iloc[0]
                        provincia_trovata = True

            if not provincia_trovata:
                print(f'  Provincia {provincia} non trovata nei dati OSM')

                strade_provincia = df_regione[df_regione['PROVINCIA'] == provincia]
                indici_da_eliminare.extend(strade_provincia.index.tolist())
                continue

            # Filtra strade per provincia
            df_provincia = df_regione[df_regione['PROVINCIA'] == provincia].copy()
            print(f'  Strade da analizzare: {len(df_provincia)}')

            # Analizza provincia
            try:
                vertici, risultati = analizza_provincia_completa(
                    provincia,
                    df_provincia,
                    provincia_geometria = poly_provincia
                )

                if vertici is None:
                    print(f"  Errore nell'analisi della provincia {provincia}")
                    continue

                # Aggiorna df
                for risultato in risultati:
                    idx = risultato['indice_originale']
                    if not risultato.get('trovata', False):
                        indici_da_eliminare.append(idx)
                        continue

                    #aggiorna info stradali
                    try:
                        mask = (
                            vertici['ref'].fillna('').str.contains(risultato['nome_pulito'], case=False, regex=True) |
                            vertici['name'].fillna('').str.contains(risultato['nome_pulito'], case= False, regex =True)
                        )

                        strDati = vertici[mask]
                        if strDati.empty:
                            print(f" Strada {risultato['nome_pulito']} non trovata in OSM-> eliminata")
                            indici_da_eliminare.append(idx)
                            continue

                        if not strDati.empty:
                            # estrai info strada
                            if 'maxspeed' in strDati.columns:
                                maxspeed_series = strDati['maxspeed'].dropna()
                                if not maxspeed_series.empty:
                                    speeds = maxspeed_series.astype(str).str.extract(r'(\d+)')[0].dropna()
                                    if not speeds.empty:
                                        df.loc[idx,'limiteVel'] =pd.to_numeric(speeds, errors='coerce').mean()
                                    else:
                                        df.loc[idx, "limiteVel"] = None
                                else:
                                    df.loc[idx, "limiteVel"] = None
                            else:
                                df.loc[idx, "limiteVel"] = None

                            if "lanes" in strDati.columns:
                                lanes_series = strDati["lanes"].dropna()
                                if not lanes_series.empty:
                                    df.loc[idx, "numeroCorsie"] = pd.to_numeric(lanes_series, errors="coerce").mean()
                                else:
                                    df.loc[idx, "numeroCorsie"] = None
                            else:
                                df.loc[idx, "numeroCorsie"] = None

                            # 3. Senso unico
                            if "oneway" in strDati.columns:
                                oneway_count = strDati["oneway"].astype(str).str.lower().isin(["yes", "true", "1"]).sum()
                                df.loc[idx, "sensoUnico"] = oneway_count
                            else:
                                df.loc[idx, "sensoUnico"] = 0

                            if "bridge" in strDati.columns:
                                bridge_count = strDati["bridge"].astype(str).str.lower().isin(["yes", "true", "1"]).sum()
                                df.loc[idx, "ponte"] = bridge_count
                            else:
                                df.loc[idx, "ponte"] = 0

                            if "junction" in strDati.columns:
                                junction_count = strDati["junction"].notna().sum()
                                df.loc[idx, "giunzione"] = junction_count

                            if "width" in strDati.columns:
                                width = strDati["width"].dropna()
                                if not width.empty:
                                    df.loc[idx, "width"] = pd.to_numeric(width, errors="coerce").mean()

                            # 5. Tunnel
                            if "tunnel" in strDati.columns:
                                tunnel_count = strDati["tunnel"].astype(str).str.lower().isin(["yes", "true", "1"]).sum()
                                df.loc[idx, "tunnel"] = int(tunnel_count)
                            else:
                                df.loc[idx, "tunnel"] = 0

                            # 6. Highway categories
                            if "highway" in strDati.columns:
                                highway_series = strDati["highway"].dropna()
                                if not highway_series.empty:
                                    categories = {
                                        'motorway': ['motorway'],
                                        'trunk': ['trunk'],
                                        'primary': ['primary'],
                                        'secondary': ['secondary'],
                                        'residential': ['residential'],
                                        'accessi': ['services','service'],
                                    }

                                    segments_with_highway = 0
                                    category_counts = {cat: 0 for cat in categories.keys()}

                                    for _, segment in strDati.iterrows():
                                        highway_raw = segment.get('highway')
                                        if is_all_na_or_empty(highway_raw):
                                            continue

                                        segments_with_highway += 1
                                        highway_str = str(highway_raw).lower()

                                        for cat, keywords in categories.items():
                                            if any(kw in highway_str for kw in keywords):
                                                category_counts[cat] += 1
                                                break

                                    if segments_with_highway > 0:
                                        for cat in categories.keys():
                                            percent = (category_counts[cat] / segments_with_highway) * 100
                                            df.loc[idx, f"highway_{cat}"] = round(percent, 2)
                    except Exception as e:
                        print(f"  Errore estrazione info strada: {e}")

                    # Aggiorna lunghezza
                    df.loc[idx, 'lunghezza_m'] = risultato['lunghezza_m']

                    # Aggiorna strutture
                    for cat in ['scuole', 'ospedali', 'polizia', 'pompieri', 'bus',
                                'supermercati', 'farmacie', 'parcheggi', 'ristoranti', 'banche']:
                        col = f'strutture_{cat}'
                        if col in risultato:
                            df.loc[idx, col] = risultato[col]

                    # Aggiorna contatori
                    totale_strade += 1
                    strade_trovate += 1
                    if sum(risultato.get(f'strutture_{cat}', 0) for cat in [
                        'scuole', 'ospedali', 'polizia', 'pompieri', 'bus',
                        'supermercati', 'farmacie', 'parcheggi', 'ristoranti', 'banche'
                    ]) > 0:
                        strade_con_strutture += 1

                # Aggiungi ai risultati completi
                risultati_completi.extend(risultati)

                # possibile modalità con checpoint usata nelle fasi iniziali di sviluppo
                #checkpoint_file = f'checkpoint_{regione}_{provincia}.csv'
                #df.to_csv(checkpoint_file, index=False)
                #print(f"  Checkpoint salvato: {checkpoint_file}")

            except Exception as e:
                print(f"  Errore analisi provincia {provincia}: {e}")
                continue

    # Rimuovi strade non trovate

    indici_da_eliminare = list(set(indici_da_eliminare))
    print(f"\nRimuovo {len(indici_da_eliminare)} strade non trovate")
    df.drop(indici_da_eliminare, inplace=True, errors='ignore')



    # Riepilogo finale
    print(f"\n{'=' * 80}")
    print("RIEPILOGO FINALE ANALISI")
    print(f"{'=' * 80}")
    print(f"Totale strade nel dataset: {len(df)}")
    print(f"Strade analizzate con successo: {strade_trovate}")
    print(f"Strade con almeno una struttura: {strade_con_strutture}")
    print(f"Strade senza strutture trovate: {strade_trovate - strade_con_strutture}")

    if strade_con_strutture > 0:
        print(f"\nDistribuzione strutture trovate:")
        categorie = ['scuole', 'ospedali', 'polizia', 'pompieri', 'bus',
                     'supermercati', 'farmacie', 'parcheggi', 'ristoranti', 'banche']

        for cat in categorie:
            col = f'strutture_{cat}'
            if col in df.columns:
                totali_cat = df[col].sum()
                strade_con_cat = (df[col] > 0).sum()
                print(f"  {cat}: {totali_cat} totali, presenti in {strade_con_cat} strade")

    print(f"\nRisultati salvati in 'risultati_finali_completi.csv'")
    return df




#visualizza correlazione tra features tramit matrice di correlazione di pearson
def matrice_correlazione(df):
    corr_matrix = df.corr(method='pearson')
    # print(corr_matrix)

    fig = plt.figure(figsize=(11, 8))

    sns.heatmap(
        corr_matrix,
    #    mask=mask,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=0.5,
        square=True,  # celle quadrate
        annot_kws={"size": 4.5}  # dimensione testo
    )
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    plt.title("Matrice di correlazione (Pearson)")

    return fig



def rendi_numeric(df):
    df["limiteVel"] = pd.to_numeric(df["limiteVel"], errors='coerce')
    df["numeroCorsie"] = pd.to_numeric(df["numeroCorsie"], errors='coerce')
    df["sensoUnico"] = pd.to_numeric(df["sensoUnico"], errors='coerce')
    df["ponte"] = pd.to_numeric(df["ponte"], errors='coerce')
    df["giunzione"] = pd.to_numeric(df["giunzione"], errors='coerce')
    df["highway_accessi"] = pd.to_numeric(df["highway_accessi"], errors='coerce')
    df["tunnel"] = pd.to_numeric(df["tunnel"], errors='coerce')
    df["width"] = pd.to_numeric(df["width"], errors='coerce')
    df["highway_motorway"] = pd.to_numeric(df["highway_motorway"], errors='coerce')
    df["highway_trunk"] = pd.to_numeric(df["highway_trunk"], errors='coerce')
    df["highway_primary"] = pd.to_numeric(df["highway_primary"], errors='coerce')
    df["highway_secondary"] = pd.to_numeric(df["highway_secondary"], errors='coerce')
    df["highway_residential"] = pd.to_numeric(df["highway_residential"], errors='coerce')
    df["lunghezza_m"] = pd.to_numeric(df["lunghezza_m"], errors='coerce')

    # inserire 0 e '' al posto di Null (controversa)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("")
    return df

#eliminiamo le colonne appartenenti al dataset originale tranne i potenziali target
#considerato che quelle informazioni non sono reperibili al momento della predizione
def pulizia_info_pregresse(df):
    df = df.drop(
        ["COD REG", "COD PROV", "REGIONE", "PROVINCIA", "CODICE", "NOME STRADA", "DA KM", "A KM", "NVEI", "Inc2RMoto",
         "IncMerci",
         "IncBici e Micromobilità", "ScFRONTALE", "ScFRONTLAT-ScLAT",
         "TAMPONAMENTO", "InvPEDONI", "FUORIUSCITA", "ALTRO", "width", "highway_accessi"], axis=1)
    return df





#iqr leggero sul dataset per togliere valori fuori scala
#probailmente presenti da errori nella creazione del dataset
def iqr_leggero(df, target = "INCIDENTI"):
    #target = "INCIDENTI"

    # Calcolo quartili sul target
    Q1 = df[target].quantile(0.25)
    Q3 = df[target].quantile(0.75)
    IQR = Q3 - Q1

    # Maschera IQR
    mask = (df[target] >= Q1 - 1.5 * IQR) & (df[target] <= Q3 + 1.5 * IQR)

    # Dataset pulito
    df_clean = df[mask]

    # Outlier rimossi (opzionale ma consigliato salvarli)
    df_outliers = df[~mask]
    return df_clean, df_outliers


#feature engineering

def add_engineered_features(df):
    EPS = 1e-3  # per evitare divisioni per zero
    lunghezzaKm = df["lunghezza_m"] / 1000

    for cat in ["motorway", "trunk", "primary"]:
        df[f"{cat}_exposure"] = df[f"highway_{cat}"] * lunghezzaKm


    df["complexity_index"] = (
            df["giunzione"] * 0.4 +
            df["ponte"] * 0.2 +
            df["tunnel"] * 0.2 +
            df["sensoUnico"] * 0.2
    )

    df["poi_totali"] = (
            df["strutture_scuole"] +
            df["strutture_ospedali"] +
            df["strutture_polizia"] +
            df["strutture_pompieri"] +
            df["strutture_bus"] +
            df["strutture_supermercati"] +
            df["strutture_farmacie"] +
            df["strutture_parcheggi"] +
            df["strutture_ristoranti"] +
            df["strutture_banche"]
    )

    df["junction_density"] = df["giunzione"] / (lunghezzaKm + EPS)

    df["attrito_stradale"] = (
        df["ponte"] + df["tunnel"] + df["sensoUnico"]
    ) / (lunghezzaKm + EPS)

    df["frammentazione"] = (
        (df["giunzione"] / (lunghezzaKm + EPS)) +
        df["attrito_stradale"]
    )

    # vulnerabilità data dalla densità di strutture molto frequenstate? + bambini
    df["vulnerabilita"] = (
        df["strutture_scuole"] * 2 +
        df["strutture_bus"] * 1.5 +
        df["strutture_supermercati"] +
        df["strutture_ristoranti"]
    ) / (lunghezzaKm + EPS)

    df["urban_density"] = (
        df["highway_residential"] +
        df["highway_secondary"]
    ) / (lunghezzaKm + EPS)

    poi_cols = [
        "strutture_scuole", "strutture_ospedali", "strutture_polizia",
        "strutture_bus", "strutture_supermercati",
        "strutture_farmacie", "strutture_parcheggi",
        "strutture_ristoranti", "strutture_banche"
    ]

    df["poi_density"] = df[poi_cols].sum(axis=1) / (lunghezzaKm + EPS)

    return df







