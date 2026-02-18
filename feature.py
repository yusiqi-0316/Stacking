import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.feature_selection import VarianceThreshold

whole_data = pd.read_csv('C:/whole.csv')
smiles_data = pd.read_csv('C:/smiles.csv')
cation_smiles = whole_data['cation'].map(smiles_data.set_index('Abbreviation')['Smiles'])
anion_smiles = whole_data['anion'].map(smiles_data.set_index('Abbreviation')['Smiles'])

def generate_morgan_fingerprints(smiles, radius=3, n_bits=2048):
    if pd.isna(smiles) or smiles.strip() == '':
        return np.zeros(n_bits)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits))

cation_fps = [generate_morgan_fingerprints(s) for s in cation_smiles]
anion_fps = [generate_morgan_fingerprints(s) for s in anion_smiles]

cation_df = pd.DataFrame(cation_fps, columns=[f'FP_cation_{i}' for i in range(2048)])
anion_df = pd.DataFrame(anion_fps, columns=[f'FP_anion_{i}' for i in range(2048)])

morgan_df = pd.concat([cation_df, anion_df], axis=1)



def filter_features_by_variance_and_correlation(df, variance_threshold=0, correlation_threshold=1.0):
    variance_filter = VarianceThreshold(threshold=variance_threshold)
    df_variance_filtered = variance_filter.fit_transform(df)
    variance_selected_features = [col for col in df.columns if col in df.columns[variance_filter.get_support()]]
    df_variance_filtered = pd.DataFrame(df_variance_filtered, columns=variance_selected_features)

    corr_matrix = df_variance_filtered.corr()
    correlated_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                colname = corr_matrix.columns[i]
                correlated_features.add(colname)
    df_filtered = df_variance_filtered.drop(columns=list(correlated_features))
    return df_filtered


filtered_morgan_df = filter_features_by_variance_and_correlation(
    morgan_df,
    variance_threshold=0,
    correlation_threshold=1.0
)

