import pandas as pd


df = pd.read_csv("data/job_offers.csv",encoding="utf-8",low_memory=False)

selected_columns = [
  "Title",
  "Description",
  "Category",
  "Skills_Tags"
]


missing = [col for col in selected_columns if col not in df.columns]

if missing:
  raise ValueError(f"column {missing} not found")

print(f"Doublons exacts dans le fichier source complet : {int(df.duplicated().sum())}")

df_clean = df[selected_columns].copy()

print("Avant nettoyage :")
for column in ["Title", "Description", "Category"]:
  null_count = int(df_clean[column].isna().sum())
  blank_count = int(df_clean[column].astype("string").str.strip().eq("").fillna(False).sum())
  print(f"  {column} : {null_count} valeurs nulles, {blank_count} valeurs vides")
print(f"  Doublons exacts sur les colonnes retenues : {int(df_clean.duplicated().sum())}")
print(f"  Titres dupliqués : {int(df_clean.duplicated(subset=['Title']).sum())}")

df_clean = df_clean.dropna(subset=["Title", "Description", "Category"])
print("\nAprès suppression des valeurs nulles :")
for column in ["Title", "Description", "Category"]:
  null_count = int(df_clean[column].isna().sum())
  blank_count = int(df_clean[column].astype("string").str.strip().eq("").fillna(False).sum())
  print(f"  {column} : {null_count} valeurs nulles, {blank_count} valeurs vides")
print(f"  Doublons exacts sur les colonnes retenues : {int(df_clean.duplicated().sum())}")
print(f"  Titres dupliqués : {int(df_clean.duplicated(subset=['Title']).sum())}")
print(f"  Nombre de lignes conservées : {len(df_clean)}")

df_clean.to_csv("data/clean_offers.csv",index=False,encoding="utf-8")