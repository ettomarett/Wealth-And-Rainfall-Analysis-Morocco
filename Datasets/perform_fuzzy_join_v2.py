import geopandas as gpd
import pandas as pd
from thefuzz import process, fuzz
import re

# File paths
geojson_path = "/home/ubuntu/dataproject/geodata/geoBoundaries-MAR-ADM2.geojson"
adm2_csv_path = "/home/ubuntu/upload/mar_adm2.csv"
output_merged_path = "/home/ubuntu/dataproject/geodata/merged_adm2_data.geojson"

def clean_name(name):
    if not isinstance(name, str):
        return ""
    # 1. Remove content in parentheses (often Arabic names or alternative spellings)
    name = re.sub(r"\(.*?\)", "", name) # Made parenthesis non-greedy
    # 2. Remove all Arabic characters (Unicode range for Arabic)
    name = re.sub(r'[\u0600-\u06FF]+', '', name)
    # 3. Convert to lowercase
    name = name.lower()
    # 4. Normalize common terms (example: "province de", "prefecture de")
    name = name.replace("province de", "").replace("province", "")
    name = name.replace("prefecture de", "").replace("prefecture", "")
    # 5. Strip leading/trailing whitespace
    name = name.strip()
    # 6. Remove extra internal spaces that might result from replacements
    name = re.sub(r"\s+", " ", name).strip()
    return name

print(f"Loading GeoJSON file: {geojson_path}")
try:
    gdf = gpd.read_file(geojson_path)
    print("GeoJSON loaded successfully.")
    gdf["shapeName_clean"] = gdf["shapeName"].apply(clean_name)
    print("Cleaned GeoJSON shapeName examples (first 5):")
    print(gdf[["shapeName", "shapeName_clean"]].head())
except Exception as e:
    print(f"Error loading or processing GeoJSON: {e}")
    gdf = None

print(f"\nLoading CSV file: {adm2_csv_path}")
try:
    df_adm2 = pd.read_csv(adm2_csv_path)
    print("ADM2 CSV loaded successfully.")
    df_adm2["ADM2_EN_clean"] = df_adm2["ADM2_EN"].apply(clean_name)
    print("Cleaned ADM2_EN examples (first 5):")
    print(df_adm2[["ADM2_EN", "ADM2_EN_clean"]].head())
except Exception as e:
    print(f"Error loading or processing ADM2 CSV: {e}")
    df_adm2 = None

if gdf is not None and df_adm2 is not None:
    print("\nAttempting fuzzy join...")
    matches = []
    choices_in_gdf = gdf["shapeName_clean"].tolist()
    choices_in_gdf = [choice if choice is not None else "" for choice in choices_in_gdf]

    matched_gdf_indices = set()
    successful_matches = 0
    unmatched_csv_rows = []

    for index, row in df_adm2.iterrows():
        query = row["ADM2_EN_clean"]
        if not query:
            unmatched_csv_rows.append(row["ADM2_EN"])
            continue

        # Prepare choices for extractOne: list of (original_gdf_idx, cleaned_name_string)
        # Filter out already matched gdf entries and empty strings
        available_choices_tuples = []
        for gdf_idx, gdf_choice_name in enumerate(choices_in_gdf):
            if gdf_idx not in matched_gdf_indices and gdf_choice_name:
                available_choices_tuples.append((gdf_idx, gdf_choice_name))
        
        if not available_choices_tuples:
            unmatched_csv_rows.append(row["ADM2_EN"])
            continue

        # Extract the strings to pass to extractOne
        current_choice_strings = [t[1] for t in available_choices_tuples]
        best_match_result = process.extractOne(query, current_choice_strings, scorer=fuzz.WRatio, score_cutoff=85)

        if best_match_result:
            # Assuming best_match_result is (match_string, score) based on previous error
            match_name_str, score = best_match_result
            
            # Find the original_gdf_idx from available_choices_tuples that corresponds to match_name_str
            original_gdf_idx = -1
            for gdf_original_idx, choice_str_in_tuple in available_choices_tuples:
                if choice_str_in_tuple == match_name_str:
                    original_gdf_idx = gdf_original_idx
                    break
            
            if original_gdf_idx != -1:
                matches.append({
                    "csv_name": row["ADM2_EN"],
                    "csv_name_clean": query,
                    "geojson_name": gdf.loc[original_gdf_idx, "shapeName"],
                    "geojson_name_clean": match_name_str, 
                    "score": score,
                    "csv_pcode": row["ADM2_PCODE"],
                    "geojson_shapeID": gdf.loc[original_gdf_idx, "shapeID"],
                    "gdf_index": original_gdf_idx
                })
                matched_gdf_indices.add(original_gdf_idx)
                successful_matches += 1
            else:
                # This case means match_name_str was not found back in available_choices_tuples, which is unexpected.
                print(f"Error: Matched name '{match_name_str}' for query '{query}' not found in available choices. Skipping.")
                unmatched_csv_rows.append(row["ADM2_EN"])
        else:
            unmatched_csv_rows.append(row["ADM2_EN"])

    match_df = pd.DataFrame(matches)
    print(f"\nFuzzy Join Results (first 10 matches):")
    if not match_df.empty:
        print(match_df.head(10))
    else:
        print("No matches found or match_df is empty.")
        
    print(f"\nTotal rows in CSV: {len(df_adm2)}")
    print(f"Successfully matched: {successful_matches}")
    print(f"Unmatched CSV ADM2_EN names (first 10): {unmatched_csv_rows[:10]}")
    print(f"Number of unmatched CSV rows: {len(unmatched_csv_rows)}")

    if successful_matches > 0 and not match_df.empty:
        gdf["temp_gdf_index"] = gdf.index
        merge_info = match_df[["csv_pcode", "gdf_index"]].drop_duplicates(subset=["gdf_index"]) # Ensure gdf_index is unique for merge
        
        merged_gdf = gdf.merge(merge_info, left_on="temp_gdf_index", right_on="gdf_index", how="inner")
        final_merged_gdf = merged_gdf.merge(df_adm2, left_on="csv_pcode", right_on="ADM2_PCODE", how="left", suffixes=("_geojson", "_csv"))
        
        print(f"\nShape of final merged GeoDataFrame: {final_merged_gdf.shape}")
        print(f"Columns in final merged GeoDataFrame: {final_merged_gdf.columns.tolist()}")
        print(f"Saving merged GeoDataFrame to: {output_merged_path}")
        try:
            final_merged_gdf.to_file(output_merged_path, driver="GeoJSON")
            print("Merged GeoDataFrame saved successfully.")
        except Exception as e:
            print(f"Error saving merged GeoDataFrame: {e}")
    else:
        print("No successful matches found or match_df is empty, merged file not created.")
else:
    print("\nCould not load one or both files. Fuzzy join cannot proceed.")

print("\nScript finished.")

