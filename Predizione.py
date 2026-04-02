import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import pandas as pd
import osmnx as ox
import geopandas as gpd
from AnalizzaStrade import add_engineered_features


#FILE dedicato alla predizione


POI_TAXONOMY = {
    "amenity": [
        "school", "hospital", "police", "fire_station",
        "pharmacy", "parking", "restaurant", "bank"
    ],
    "shop": [
        "supermarket", "shop"
    ],
    "highway": [
        "bus_stop"
    ]
}


def analyze_route_environment(
        graph,
        route,
        buffer_m=100,
        poi_tags=None
):
    #analizza la strada per effettuare la predizione

    print("\n" + "=" * 50)
    print("ANALISI AREA PERCORSO")
    print("=" * 50)

    if poi_tags is None:
        poi_tags = {
            "amenity": ["school", "hospital", "police", "fire_station",
                        "pharmacy", "parking", "restaurant", "bank"],
            "shop": ["supermarket", "shop"],
            "highway": ["bus_stop"]
        }

    # Percorso -> GeoDataFrame
    print("1. Estrazione geometria percorso...")
    try:
        route_gdf = ox.routing.route_to_gdf(graph, route)
        if route_gdf.empty:
            print("  Nessuna geometria del percorso trovata")
            return None, None, None
    except Exception as e:
        print(f"  Errore estrazione percorso: {e}")
        # Fallback: crea route_gdf manualmente
        route_coords = []
        for node in route:
            if node in graph.nodes:
                route_coords.append((graph.nodes[node]['x'], graph.nodes[node]['y']))

        if not route_coords:
            return None, None, None

        from shapely.geometry import LineString
        route_line = LineString(route_coords)
        route_gdf = gpd.GeoDataFrame(
            geometry=[route_line],
            crs="EPSG:4326"
        )

    print(f"    Percorso: {len(route_gdf)} segmenti")

    #  Proiezione metrica corretta
    print("2. Creazione buffer area...")

    route_proj = ox.projection.project_gdf(route_gdf)
    projected_crs = route_proj.crs

    buffer_geom = route_proj.geometry.buffer(buffer_m)
    buffer_union = buffer_geom.union_all()

    buffer_wgs84 = (
        gpd.GeoSeries([buffer_union], crs=projected_crs)
        .to_crs("EPSG:4326")
        .iloc[0]
    )

    print(f"   Buffer creato: {buffer_m}m dal percorso")

    #  STRUTTURA STRADA nell'area
    print("3. Analisi rete stradale...")
    try:
        # Estrai tutti gli archi del grafo
        edges_gdf = ox.graph_to_gdfs(graph, nodes=False)

        # Filtra archi dentro il buffer
        edges_in_buffer = edges_gdf[edges_gdf.intersects(buffer_wgs84)].copy()

        if edges_in_buffer.empty:
            print("     Nessuna strada trovata nel buffer")
            road_stats = {}
        else:
            # Proietta per calcoli metrici
            edges_proj = edges_in_buffer.to_crs(projected_crs)
            # edges_metric = edges_in_buffer.to_crs(epsg_code) if 'epsg_code' in locals() else edges_in_buffer
            road_stats = {
                "lunghezza_totale_m": edges_proj.length.sum(),
                "numero_segmenti": len(edges_proj),
            }

            # ---------- MAXSPEED ----------
            if "maxspeed" in edges_in_buffer.columns:
                speeds = (
                    edges_in_buffer["maxspeed"]
                    .dropna()
                    .astype(str)
                    .str.extract(r"(\d+)")[0]
                    .astype(float)
                )
                road_stats["limiteVel"] = speeds.mean()
            else:
                road_stats["limiteVel"] = np.nan

            # ---------- LANES ----------
            if "lanes" in edges_in_buffer.columns:
                lanes = pd.to_numeric(edges_in_buffer["lanes"], errors="coerce")
                road_stats["numeroCorsie"] = lanes.mean()
            else:
                road_stats["numeroCorsie"] = np.nan

            # ---------- ONEWAY ----------
            if "oneway" in edges_in_buffer.columns:
                oneway_bool = edges_in_buffer["oneway"].astype(str).str.lower().isin(["yes", "true", "1"])
                road_stats["sensoUnico_ratio"] = oneway_bool.mean()
            else:
                road_stats["sensoUnico_ratio"] = 0.0

            # ---------- PONTI ----------
            if "bridge" in edges_in_buffer.columns:
                road_stats["ponte"] = (
                    edges_in_buffer["bridge"]
                    .astype(str)
                    .str.lower()
                    .isin(["yes", "viaduct", "true"])
                    .sum()
                )
            else:
                road_stats["ponte"] = 0

            # ---------- TUNNEL ----------
            if "tunnel" in edges_in_buffer.columns:
                road_stats["tunnel"] = (
                    edges_in_buffer["tunnel"]
                    .astype(str)
                    .str.lower()
                    .isin(["yes", "true"])
                    .sum()
                )
            else:
                road_stats["tunnel"] = 0

            # ---------- JUNCTION ----------
            if "junction" in edges_in_buffer.columns:
                road_stats["giunzione"] = edges_in_buffer["junction"].notna().sum()
            else:
                road_stats["giunzione"] = 0

            # ---------- HIGHWAY TYPES (percentuali) ----------
            if "highway" in edges_in_buffer.columns:
                highway = edges_in_buffer["highway"].astype(str).str.lower()
                total = len(highway)

                for cat in ["motorway", "trunk", "primary", "secondary", "residential"]:
                    road_stats[f"highway_{cat}"] = (highway.str.contains(cat).sum() / total) * 100
            else:
                for cat in ["motorway", "trunk", "primary", "secondary", "residential"]:
                    road_stats[f"highway_{cat}"] = 0.0


            print(f"    Strade trovate: {road_stats['numero_segmenti']} segmenti")
            print(f"    Lunghezza totale: {road_stats['lunghezza_totale_m']:.0f}m")

    except Exception as e:
        print(f"  Errore analisi strade: {e}")
        road_stats = {}

    # PUNTI DI INTERESSE nell'area
    print("4. Ricerca punti di interesse...")

    try:
        pois_gdf = ox.features_from_polygon(
            buffer_wgs84,
            tags=POI_TAXONOMY
        )

        if pois_gdf.empty:
            print("     Nessun POI trovato")
            pois_gdf = gpd.GeoDataFrame()
        else:
            # normalizza
            pois_gdf = pois_gdf.reset_index(drop=True)

            # assegna categoria principale
            def infer_category(row):
                for key, values in POI_TAXONOMY.items():
                    if key in row and row[key] in values:
                        return key
                return "other"

            pois_gdf["poi_category"] = pois_gdf.apply(infer_category, axis=1)

            # nome POI
            pois_gdf["poi_name"] = (
                pois_gdf["name"]
                if "name" in pois_gdf.columns
                else "Senza nome"
            )

            print(f"    POI trovati: {len(pois_gdf)}")

    except Exception as e:
        print(f"  Errore ricerca POI: {e}")
        pois_gdf = gpd.GeoDataFrame()

    return route_gdf, road_stats, pois_gdf


def get_optimal_route(start_coords, end_coords, network_type='drive'):
    # 1. CALCOLA DISTANZA
    distance_m = ox.distance.great_circle(
        start_coords[0], start_coords[1],  # OSMnx v1.x usa (lat1, lon1, lat2, lon2)
        end_coords[0], end_coords[1]
    )

    minDist = 300
    print(f" Distanza in linea d'aria: {distance_m:.0f} metri ({distance_m / 1000:.1f} km)")

    # 2. DECIDI la STRATEGIA
    custom_filter = None
    if distance_m < 1000:
        print("-Percorso CORTO")
        dist_buffer = max(distance_m * 3, minDist)
    elif distance_m < 10000:
        print("-Percorso MEDIO")
        dist_buffer = distance_m * 2
    else:
        print("-Percorso LUNGO - Uso solo strade principali")
        custom_filter = ('["highway"~"motorway|trunk|primary|secondary|'
                         'motorway_link|trunk_link|primary_link|secondary_link|tertiary"]')
        dist_buffer = distance_m * 1.5

    center_lat = (start_coords[0] + end_coords[0]) / 2
    center_lon = (start_coords[1] + end_coords[1]) / 2

    # 4. SCARICA RETE
    try:
        graph = ox.graph_from_point(
            (center_lat, center_lon),
            dist=dist_buffer,
            network_type=network_type,
            simplify=True,
            custom_filter=custom_filter
        )
    except Exception as e:
        print(f" Errore: {e}. Riprovo con raggio standard...")
        graph = ox.graph_from_point((center_lat, center_lon), dist=minDist * 2, network_type=network_type)

    print(f" Rete: {len(graph.nodes)} nodi")

    # 5. TROVA NODI PIÙ VICINI
    start_node = ox.distance.nearest_nodes(graph, X=start_coords[1], Y=start_coords[0])
    end_node = ox.distance.nearest_nodes(graph, X=end_coords[1], Y=end_coords[0])

    # 6. CALCOLA PERCORSO
    try:
        route = ox.shortest_path(graph, start_node, end_node, weight='length')

        if route:
            # Usiamo routing.route_to_gdf per ottenere i dati del percorso in modo sicuro
            route_gdf = ox.routing.route_to_gdf(graph, route, weight='length')
            length = route_gdf['length'].sum()

            # grafico
            print(" Generazione mappa in corso...")
            fig, ax = ox.plot_graph_route(
                graph,
                route,
                route_color='red',
                route_linewidth=3,
                node_size=0,
                bgcolor='white',
                edge_color='#999999',
                edge_linewidth=0.5,
                show=False,
                close=False
            )

            # Aggiungiamo i punti di partenza e arrivo come pallini sopra la mappa
            ax.scatter(start_coords[1], start_coords[0], c='green', s=100, label='Start', zorder=5)
            ax.scatter(end_coords[1], end_coords[0], c='blue', s=100, label='End', zorder=5)
            plt.legend()

            #DE-COMMENTARE SE SI VUOLE STAMPARE il TRAGITTO a schermo ad ogni PREDIZIONE
            #plt.show()

            print(f" PERCORSO TROVATO: {length:.0f} m")
            return route, graph, length
    except Exception as e:
        print(f" ERRORE nel calcolo percorso: {e}")
        return None, None, 0


def df_predizione_utente(nome, start, end):

    ox.settings.use_cache = True

    route, graph, length = get_optimal_route(start, end)
    route_gdf, stat_strada, pois_gdf = analyze_route_environment(graph, route, buffer_m=100)
    print(f'\n ==== {stat_strada}')
    if "amenity" in pois_gdf.columns:
        stat_amenity = (pois_gdf.dropna(subset=["amenity"]).groupby("amenity").size().to_dict())
    else:
        stat_amenity = {}

    if "shop" in pois_gdf.columns:
        stat_shop = (pois_gdf.dropna(subset=["shop"]).groupby("shop").size().to_dict())
    else:
        stat_shop = {}

    if "highway" in pois_gdf.columns:
        stat_highway = (pois_gdf.dropna(subset=["highway"]).groupby("highway").size().to_dict())
    else:
        stat_highway = {}
    print(f'=== {stat_amenity}')
    print(f'=== {stat_shop}')
    print(f'=== {stat_highway}')

    df_strada_utente = pd.DataFrame([{
        "limiteVel": stat_strada.get("limiteVel"),
        "numeroCorsie": stat_strada.get("numeroCorsie"),
        "sensoUnico": stat_strada.get("sensoUnico_ratio"),
        "ponte": stat_strada.get("ponte"),
        "giunzione": stat_strada.get("giunzione"),
        "tunnel": stat_strada.get("tunnel"),

        "highway_motorway": stat_strada.get("highway_motorway"),
        "highway_trunk": stat_strada.get("highway_trunk"),
        "highway_primary": stat_strada.get("highway_primary"),
        "highway_secondary": stat_strada.get("highway_secondary"),
        "highway_residential": stat_strada.get("highway_residential"),

        "strutture_scuole": stat_amenity.get("school", 0),
        "strutture_ospedali": stat_amenity.get("hospital", 0),
        "strutture_polizia": stat_amenity.get("police", 0),
        "strutture_pompieri": stat_amenity.get("fire_station", 0),
        "strutture_bus": stat_highway.get("bus_stop", 0),
        "strutture_supermercati": stat_shop.get("supermarket", 0) + stat_shop.get("shop", 0),
        "strutture_farmacie": stat_amenity.get("pharmacy", 0),
        "strutture_parcheggi": stat_amenity.get("parking", 0),
        "strutture_ristoranti": stat_amenity.get("restaurant", 0),
        "strutture_banche": stat_amenity.get("bank", 0),

        "lunghezza_m": length}])

    if route:
        print(f" Efficienza: {len(graph.nodes)} nodi per {length:.0f}m")

    df_strada_utente = add_engineered_features(df_strada_utente)
    return df_strada_utente

def effettua_predizione(nome, start, end, df_completo):
    contenuto = joblib.load("modello/modelloGB_finale.joblib")

    modello = contenuto["modello"]
    df_percorso = df_predizione_utente(nome, start, end)
    df_percorso = df_percorso[contenuto["feature"]]
    predizione = modello.predict(df_percorso)
    incidentiPerKmPredetti = predizione / (df_percorso["lunghezza_m"].iloc[0] / 1000)

    print(f"\n\nIncidenti annuali predetti: {predizione} (implicando una cadenza: {incidentiPerKmPredetti} Inc/km)")

    print(f"Incidenti medi: {(df_completo["INCIDENTI"] / (df_completo["lunghezza_m"] / 1000).replace(0, np.nan)).median()} Inc/km")