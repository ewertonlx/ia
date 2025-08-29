import pandas as pd

def read_csv(file):
    chunks = pd.read_csv(file, chunksize=100000, sep='\t', low_memory=False)
    df = next(chunks)
    
    cols_to_keep = [
        'code', 'product_name', 'nutrition-score-fr_100g', 'quantity', 
        'fruits-vegetables-nuts_100g', 'fruits-vegetables-nuts-estimate_100g', 
        'collagen-meat-protein-ratio_100g', 'cocoa_100g', 
        'chlorophyl_100g', 'carbon-footprint_100g', 
        'glycemic-index_100g', 'water-hardness_100g'
    ]

    df = df[cols_to_keep]
    df = df.dropna(subset=['quantity', 'nutrition-score-fr_100g'])

    cols_for_dummies = [
        'quantity',
        'fruits-vegetables-nuts_100g', 
        'fruits-vegetables-nuts-estimate_100g', 
        'collagen-meat-protein-ratio_100g', 
        'cocoa_100g', 
        'chlorophyl_100g', 
        'carbon-footprint_100g', 
        'glycemic-index_100g', 
        'water-hardness_100g'
    ]
    df_encoded = pd.get_dummies(df, columns=cols_for_dummies, dtype=int)
    return df_encoded