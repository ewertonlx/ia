CREATE TABLE IF NOT EXISTS sor_food (
    code VARCHAR(255) PRIMARY KEY,
    product_name TEXT,
    nutrition_score_fr_100g REAL,
    quantity TEXT,
    fruits_vegetables_nuts_100g REAL,
    fruits_vegetables_nuts_estimate_100g REAL,
    collagen_meat_protein_ratio_100g REAL,
    cocoa_100g REAL,
    chlorophyl_100g REAL,
    carbon_footprint_100g REAL,
    glycemic_index_100g REAL,
    water_hardness_100g REAL
);