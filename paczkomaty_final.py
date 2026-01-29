import geopandas as gpd
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point
import pathlib
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ======================================================
# 1. KONFIGURACJA I PARAMETRY
# ======================================================

CRS_WGS = "EPSG:4326"
CRS_METRIC = "EPSG:2180"  # PUWG 1992 dla Polski (metry)

WALK_SPEED_KMH = 4.8
GRID_SIZE = 50          # Rozdzielczość siatki kandydatów (m)
TOP_N = 3                # Ile lokalizacji wybrać
MIN_DIST_OWN = 300       # Min. odległość od własnych paczkomatów (kanibalizacja)

WEIGHTS = {
    "shops": 3,
    "schools": 2,
    "kindergartens": 2,
    "offices": 4,
    "residential": 5,
    "stops": 3,
    "fuel": 1
}

COMPETITION_RANGE = 250  
COMPETITION_BONUS = 5    

# ======================================================
# 2. OBSZAR ANALIZY I SIECI
# ======================================================

print("▶ Wczytywanie granicy obszaru...")
border = gpd.read_file("projekt_2_granice/strefa wielkomiejska_bufor_200.shp").to_crs(CRS_WGS)
border["geometry"] = border.geometry.make_valid()
border = border.explode(ignore_index=True)
border["area"] = border.geometry.area
border = border.sort_values("area", ascending=False).iloc[[0]].drop(columns="area")

print("▶ Pobieranie i budowa sieci pieszej OSM...")
G = ox.graph_from_polygon(
    border.geometry.iloc[0],
    network_type="walk",
    simplify=True
)
G = ox.project_graph(G, to_crs=CRS_METRIC)

speed_mps = WALK_SPEED_KMH * 1000 / 3600
for u, v, k, data in G.edges(keys=True, data=True):
    data["time"] = data.get("length", 0) / speed_mps

# ======================================================
# 3. GENERATORY RUCHU I MASKI
# ======================================================

print("▶ Wczytywanie generatorów ruchu...")
shops = gpd.read_file("generated_data/shops.gpkg").to_crs(CRS_METRIC)
schools = gpd.read_file("generated_data/schools.gpkg").to_crs(CRS_METRIC)
kindergartens = gpd.read_file("generated_data/kindergartens.gpkg").to_crs(CRS_METRIC)
offices = gpd.read_file("generated_data/offices.gpkg").to_crs(CRS_METRIC)

print("▶ Pobieranie danych uzupełniających z OSM...")
residential = ox.features_from_polygon(
    border.geometry.iloc[0],
    tags={"building": ["apartments", "residential", "house"], "landuse": "residential"}
).to_crs(CRS_METRIC)

stops = ox.features_from_polygon(
    border.geometry.iloc[0],
    tags={"highway": "bus_stop", "public_transport": "platform", "railway": ["tram_stop", "station"]}
).to_crs(CRS_METRIC)

fuel = ox.features_from_polygon(
    border.geometry.iloc[0],
    tags={"amenity": "fuel"}
).to_crs(CRS_METRIC)

generators = {
    "shops": shops,
    "schools": schools,
    "kindergartens": kindergartens,
    "offices": offices,
    "residential": residential,
    "stops": stops,
    "fuel": fuel
}

print("▶ Tworzenie maski wykluczeń...")
mask_tags = {
    "landuse": ["industrial", "cemetery"],
    "natural": ["water", "wetland"],
    "leisure": ["park", "pitch"],
    "railway": True
}
masks = ox.features_from_polygon(border.geometry.iloc[0], tags=mask_tags)
mask_geom = masks.to_crs(CRS_METRIC).unary_union

# ======================================================
# 4. PRZETWARZANIE PACZKOMATÓW
# ======================================================

print("▶ Klasyfikacja paczkomatów...")
all_lockers = ox.features_from_polygon(
    border.geometry.iloc[0],
    tags={"amenity": "parcel_locker"}
).to_crs(CRS_METRIC)

for col in ['brand', 'operator', 'name']:
    if col not in all_lockers.columns:
        all_lockers[col] = ''

inpost_query = "InPost|Paczkomat"
is_inpost = (
    all_lockers['operator'].str.contains(inpost_query, case=False, na=False) |
    all_lockers['brand'].str.contains(inpost_query, case=False, na=False) |
    all_lockers['name'].str.contains(inpost_query, case=False, na=False)
)

inpost_lockers = all_lockers[is_inpost]
competitor_lockers = all_lockers[~is_inpost]

def get_nodes(gdf):
    if gdf.empty: return []
    return list(set(ox.distance.nearest_nodes(G, gdf.geometry.centroid.x, gdf.geometry.centroid.y)))

inpost_nodes = get_nodes(inpost_lockers)
comp_nodes = get_nodes(competitor_lockers)

# ======================================================
# 5. OBLICZENIA SIECIOWE
# ======================================================

print("▶ Pre-kalkulacja dostępności generatorów...")
generator_access_maps = {}

# Zwykła pętla po słowniku z tqdm działa zawsze
for name, gdf in tqdm(generators.items(), desc="Generatory"):
    if gdf.empty: continue
    nodes = ox.distance.nearest_nodes(G, gdf.geometry.centroid.x, gdf.geometry.centroid.y)
    nodes = list(set(nodes))
    dists = nx.multi_source_dijkstra_path_length(G, nodes, weight="time")
    generator_access_maps[name] = dists

# ======================================================
# 5.5 PRE-KALKULACJA ODLEGŁOŚCI DO PACZKOMATÓW
# ======================================================

print("▶ Pre-kalkulacja odległości do paczkomatów InPost...")
if inpost_nodes:
    inpost_dists = nx.multi_source_dijkstra_path_length(G, inpost_nodes, weight="length")
else:
    inpost_dists = {}

print("▶ Pre-kalkulacja odległości do konkurencji...")
if comp_nodes:
    comp_dists = nx.multi_source_dijkstra_path_length(G, comp_nodes, weight="length")
else:
    comp_dists = {}

# ======================================================
# 6. GENEROWANIE KANDYDATÓW
# ======================================================

print("▶ Generowanie siatki kandydatów...")
border_m = border.to_crs(CRS_METRIC)
minx, miny, maxx, maxy = border_m.total_bounds

points = []
x_range = np.arange(minx, maxx, GRID_SIZE)
y_range = np.arange(miny, maxy, GRID_SIZE)

with tqdm(total=len(x_range) * len(y_range), desc="Tworzenie siatki") as pbar:
    for x in x_range:
        for y in y_range:
            p = Point(x, y)
            if border_m.contains(p).iloc[0] and not p.intersects(mask_geom):
                points.append(p)
            pbar.update(1)

candidates = gpd.GeoDataFrame(geometry=points, crs=CRS_METRIC)
print("▶ Mapowanie kandydatów do sieci...")
# Tu nie trzeba paska postępu, nearest_nodes jest szybkie wektorowo
candidates["node"] = ox.distance.nearest_nodes(G, candidates.geometry.x, candidates.geometry.y)
print(f"✔ Liczba punktów do oceny: {len(candidates)}")

# ======================================================
# 7. FUNKCJA OCENY
# ======================================================

def calculate_score(row_node, max_minutes, scenario="basic"):
    max_seconds = max_minutes * 60
    score = 0
    
    for gen_name, access_map in generator_access_maps.items():
        dist = access_map.get(row_node, np.inf)
        if dist <= max_seconds:
            score += WEIGHTS[gen_name]

    if scenario == "competition" and comp_dists:
        if score > 0:
            min_dist = comp_dists.get(row_node, np.inf)
            if min_dist <= COMPETITION_RANGE:
                score += COMPETITION_BONUS

    return score

def calculate_cannibalization(n):
    if not inpost_dists:
        return np.inf
    return inpost_dists.get(n, np.inf)

# ======================================================
# 8. WYKONANIE ANALIZY (BEZPIECZNA METODA Z TQDM)
# ======================================================

print("\n▶ ROZPOCZYNAM GŁÓWNE OBLICZENIA...")

# ZAMIAST progress_apply UŻYWAMY LIST COMPREHENSION Z TQDM
# To omija błąd biblioteki pandas

print("   Krok 0/5: Weryfikacja kanibalizacji...")
candidates["dist_to_own"] = [
    calculate_cannibalization(n) 
    for n in tqdm(candidates["node"], desc="Kanibalizacja")
]

valid_candidates = candidates[candidates["dist_to_own"] >= MIN_DIST_OWN].copy()
# Reset indexu jest ważny po filtrowaniu, żeby iteracja się zgadzała
valid_candidates = valid_candidates.reset_index(drop=True)

print(f"   Pozostało kandydatów: {len(valid_candidates)}")

# Funkcje pomocnicze do lambdy w pętli
nodes_list = valid_candidates["node"].tolist()

print("\n   Krok 1/5: Scenariusz 1 (3 minuty)...")
valid_candidates["score_s1_3m"] = [
    calculate_score(n, 3, "basic") for n in tqdm(nodes_list, desc="S1 3min")
]

print("\n   Krok 2/5: Scenariusz 1 (8 minut)...")
valid_candidates["score_s1_8m"] = [
    calculate_score(n, 8, "basic") for n in tqdm(nodes_list, desc="S1 8min")
]

print("\n   Krok 3/5: Scenariusz 2 (3 min z konkurencją)...")
valid_candidates["score_s2_3m"] = [
    calculate_score(n, 3, "competition") for n in tqdm(nodes_list, desc="S2 3min")
]

print("\n   Krok 4/5: Scenariusz 2 (8 min z konkurencją)...")
valid_candidates["score_s2_8m"] = [
    calculate_score(n, 8, "competition") for n in tqdm(nodes_list, desc="S2 8min")
]

# ======================================================
# 9. WYBÓR NAJLEPSZYCH LOKALIZACJI
# ======================================================

def select_top_locations(df, score_col):
    sorted_df = df.sort_values(score_col, ascending=False)
    selected = []
    
    for _, row in sorted_df.iterrows():
        if not selected:
            selected.append(row)
        else:
            is_far_enough = all(row.geometry.distance(s.geometry) > MIN_DIST_OWN for s in selected)
            if is_far_enough:
                selected.append(row)
        if len(selected) == TOP_N:
            break  
    return gpd.GeoDataFrame(selected, crs=CRS_METRIC)

print("\n▶ Wybieranie najlepszych lokalizacji...")

results = {
    "scenariusz1_3min": select_top_locations(valid_candidates, "score_s1_3m"),
    "scenariusz1_8min": select_top_locations(valid_candidates, "score_s1_8m"),
    "scenariusz2_3min": select_top_locations(valid_candidates, "score_s2_3m"),
    "scenariusz2_8min": select_top_locations(valid_candidates, "score_s2_8m"),
}

# ======================================================
# 10. ZAPIS WYNIKÓW
# ======================================================

pathlib.Path("results").mkdir(exist_ok=True)
valid_candidates.drop(columns=["node"]).to_file("results/all_candidates_scored.gpkg", driver="GPKG")

for name, gdf in results.items():
    if not gdf.empty:
        gdf.to_file(f"results/best_{name}.gpkg", driver="GPKG")
        print(f"✔ Zapisano: results/best_{name}.gpkg")
        print(f"--- {name} ---")
        print(gdf[["score_s1_3m", "score_s1_8m", "score_s2_3m", "score_s2_8m"]].to_string())
        print("")

print("Zakończono!")