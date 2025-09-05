import pandas as pd

def read_csv(file, top_n_quantity=20):
    chunks = pd.read_csv(file, chunksize=100000, sep='\t', low_memory=False)
    df = next(chunks)

    cols_to_keep = [
        'code', 'product_name', 'nutrition-score-fr_100g', 'quantity', 
        'fruits-vegetables-nuts_100g', 'fruits-vegetables-nuts-estimate_100g', 
        'collagen-meat-protein-ratio_100g', 'cocoa_100g', 
        'chlorophyl_100g', 'carbon-footprint_100g', 
        'glycemic-index_100g', 'water-hardness_100g'
    ]
    df = df[[c for c in cols_to_keep if c in df.columns]]
    df = df.dropna(subset=['quantity', 'nutrition-score-fr_100g'])

    if 'quantity' in df.columns:
        top_values = df['quantity'].value_counts().nlargest(top_n_quantity).index
        df['quantity'] = df['quantity'].apply(lambda x: x if x in top_values else 'Other')

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
    existing_cols = [c for c in cols_for_dummies if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=existing_cols, dtype=int)

    df_encoded = df_encoded.loc[:, ~df_encoded.columns.duplicated()]
    df_encoded.columns = df_encoded.columns.str.strip().str.replace(r"[^\w]", "_", regex=True)

    return df_encoded