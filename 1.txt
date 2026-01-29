import geopandas as gpd
import requests
import pathlib

# read borders
border = gpd.read_file("projekt_2_granice/strefa wielkomiejska_bufor_200.shp").to_crs("EPSG:4326")

# check if osm_data folder exists
osm_data_folder = "osm_data"
pathlib.Path(osm_data_folder).mkdir(parents=True, exist_ok=True)
generated_data_folder = "generated_data"
pathlib.Path(generated_data_folder).mkdir(parents=True, exist_ok=True)

# check if lodzkie-latest.osm.pbf exists
if not pathlib.Path("lodzkie-latest.osm.pbf").exists():
    # download the file
    url = "https://download.geofabrik.de/europe/poland/lodzkie-latest.osm.pbf"
    r = requests.get(url, allow_redirects=True)
    with open("lodzkie-latest.osm.pbf", "wb") as f:
        f.write(r.content)

# read the data
for idx, layer in gpd.list_layers("lodzkie-latest.osm.pbf").iterrows():
    name = layer["name"]
    print(name)
    name_to_save = f"{osm_data_folder}/{name}.gpkg"
    # if exists, skip
    if pathlib.Path(name_to_save).exists():
        continue
    df = gpd.read_file("lodzkie-latest.osm.pbf", layer=name)
    # cut to the border and save as name.gpkg
    cut = df.clip(border).to_crs("EPSG:2180")
    cut.to_file(name_to_save)

# dodatkowo, jeśli nie istnieje points.gpkg, a istnieje points.gpkg w osm_data, zapisz warstwę 'points'
points_gpkg_path = f"{osm_data_folder}/points.gpkg"
if not pathlib.Path(points_gpkg_path).exists():
    layers = [layer["name"] for idx, layer in gpd.list_layers("lodzkie-latest.osm.pbf").iterrows()]
    if "points" in layers:
        df_points = gpd.read_file("lodzkie-latest.osm.pbf", layer="points")
        df_points.clip(border).to_crs("EPSG:2180").to_file(points_gpkg_path)

# read points and get shops
points = gpd.read_file(f"{osm_data_folder}/points.gpkg")
# save unique other_tags to csv
points["other_tags"].drop_duplicates().to_csv(f"{generated_data_folder}/other_tags.csv", index=False)
# shops are points that contain the string \"shop\"=> in other_tags
# change type to string
points["other_tags"] = points["other_tags"].astype(str)
shops = points[points["other_tags"].str.contains('"shop"=>')]
# save as shops.gpkg
shops.to_file(f"{generated_data_folder}/shops.gpkg")
schools = points[points["other_tags"].str.contains('"school"')]
schools.to_file(f"{generated_data_folder}/schools.gpkg")
kindergartens = points[points["other_tags"].str.contains('"kindergarten"')]
kindergartens.to_file(f"{generated_data_folder}/kindergartens.gpkg")
offices = points[points["other_tags"].str.contains('"office"')]
offices.to_file(f"{generated_data_folder}/offices.gpkg")

