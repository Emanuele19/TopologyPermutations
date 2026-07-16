import pandas as pd
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Set

def process_edge_file(input_tsv_path: Path, output_csv_path: Path, excluded_nodes: set[str] = None):
    """
    Processes files containing graph's edge info
    
    Params:
    - input_tsv_path: Path - expects a Path object pointing to a tsv file with the following columns:
        [node1, node2, node1_string_id, node2_string_id, homology, experimental_data_interaction, database_annoted,
        automatic_textmining, combined_score]
    - output_csv_path: Path - expects a Path object with the name of the output file

    Every field of the input file are copied to the output file, except for node1_string_id and node2_string_id.
    Those are merged into a field named "name" formatted as "node1_string_id (interacts with) node2_string_id"
    """
    df = pd.read_csv(input_tsv_path, sep='\t')
    if excluded_nodes:
        df = df[~df['node1_string_id'].isin(excluded_nodes) & ~df['node2_string_id'].isin(excluded_nodes)]
    df['name'] = df['node1_string_id'] + " (interacts with) " + df['node2_string_id']
    df = df.drop(columns=['node1_string_id', 'node2_string_id'])
    df.to_csv(output_csv_path, index=False)


def process_node_file(input_tsv_path: Path, output_csv_path: Path, excluded_nodes: set[str] = None):
    """
    Processes files containing graph's node info
    
    Params:
    - input_tsv_path: Path - expects a Path object pointing to a tsv file with the following columns:
        [Entry, Entry Name, Protein names, Gene Names, Gene Ontology (biological process), 
        Gene Ontology (cellular component), Gene Ontology (molecular function), Gene Ontology IDs, Gene Ontology (GO),
        STRING]
    - output_csv_path: Path - expects a Path object with the name of the output file

    Every field of the input file are copied to the output file, but STRING is renamed to name
    """
    df = pd.read_csv(input_tsv_path, sep='\t')
    if excluded_nodes:
        df = df[~df['Entry'].isin(excluded_nodes)]

    # Delete entries where "STRING" field is empty
    df = df.dropna(subset=['STRING'])

    # remove trailing ";" from STRING field
    df['STRING'] = df['STRING'].str.rstrip(';')


    df = df.rename(columns={'STRING': 'name'})
    df.to_csv(output_csv_path, index=False)


def run_tests():
    """
    Checks that the previous two functions work properly by comparing the outputs with the files
    AD_edges.csv, AD_nodes.csv, PD_edges.csv and PD_nodes.csv.
    This functions logs what columns are in common and checks if the values of those are equal (only on common columns)
    """
    base_path = Path("./networks")
    pairs = [
        ("AD_edges_test.csv", "AD_edges.csv"),
        ("AD_nodes_test.csv", "AD_nodes.csv"),
        ("PD_edges_test.csv", "PD_edges.csv"),
        ("PD_nodes_test.csv", "PD_nodes.csv")
    ]

    for test_name, ref_name in pairs:
        test_path, ref_path = base_path / test_name, base_path / ref_name
        if not test_path.exists() or not ref_path.exists():
            print(f"[-] Saltato test per {test_name}: file non trovato.")
            continue

        df_test = pd.read_csv(test_path)
        df_ref = pd.read_csv(ref_path)

        print(f"[#] Confronto {test_name} <-> {ref_name}")

        common_cols = sorted(list(set(df_test.columns) & set(df_ref.columns)))
        
        # Identifica le colonne chiave per l'allineamento delle righe
        if "Entry" in common_cols:
            keys = ["Entry"]
        elif "#node1" in common_cols and "node2" in common_cols:
            keys = ["#node1", "node2"]
        else:
            keys = []

        # Ordina entrambi i dataframe per garantire un confronto coerente
        if keys:
            df_test = df_test.sort_values(by=keys).reset_index(drop=True)
            df_ref = df_ref.sort_values(by=keys).reset_index(drop=True)

        if df_test.shape[0] != df_ref.shape[0]:
            print(f"    [!] Attenzione: numero di righe differente ({len(df_test)} vs {len(df_ref)})")
            if keys:
                # Eseguiamo un merge interno per confrontare solo i record presenti in entrambi
                merged = pd.merge(df_test, df_ref, on=keys, suffixes=('_test', '_ref'))
                print(f"    [i] Allineamento su chiavi comuni: confrontando {len(merged)} righe corrispondenti.")
                df_t_comp = merged[[c + "_test" for c in common_cols if c not in keys]].rename(columns=lambda x: x[:-5])
                df_r_comp = merged[[c + "_ref" for c in common_cols if c not in keys]].rename(columns=lambda x: x[:-4])
                comp_cols = [c for c in common_cols if c not in keys]
            else:
                df_t_comp, df_r_comp, comp_cols = df_test[common_cols], df_ref[common_cols], common_cols
        else:
            df_t_comp, df_r_comp, comp_cols = df_test[common_cols], df_ref[common_cols], common_cols

        mismatched_cols = []
        for col in comp_cols:
            # Gestione confronto numerico con tolleranza (per combined_score e interaction scores)
            if pd.api.types.is_numeric_dtype(df_t_comp[col]) and pd.api.types.is_numeric_dtype(df_r_comp[col]):
                # Consideriamo uguali se la differenza è minima (precisione float)
                diff_mask = ~np.isclose(df_t_comp[col].values, df_r_comp[col].values, equal_nan=True, atol=1e-4)
            else:
                # Identifica dove i valori differiscono, escludendo i casi dove entrambi sono NaN
                diff_mask = df_t_comp[col].ne(df_r_comp[col]) & ~(df_t_comp[col].isna() & df_r_comp[col].isna())
            
            mismatch_count = diff_mask.sum()
            
            if mismatch_count == 0:
                print(f"    [OK]   Colonna: {col}")
            else:
                mismatched_cols.append(col)
                print(f"    [FAIL] Colonna: {col} ({mismatch_count} celle diverse)")
                # Troubleshooting: stampa un esempio della prima differenza trovata
                idx = diff_mask.idxmax()
                print(f"           Esempio: test='{df_t_comp.loc[idx, col]}' vs rif='{df_r_comp.loc[idx, col]}'")

        if not mismatched_cols:
            print(f"    ==> RISULTATO: TEST SUPERATO")
        else:
            print(f"    ==> RISULTATO: TEST FALLITO. Colonne con discrepanze: {mismatched_cols}")

COMMON_NODES = {'P06702', 'Q13501', 'P10636', 'P49768', 'P48023', 
 'P55290', 'Q99497', 'P68036', 'P09429', 'Q00535', 
 'P20700'}

def main():
    base_sources_path = Path("./sources/test_db")
    base_networks_path = Path("./networks")

    d1_string_edge_input = base_sources_path / "STRING_FTD.tsv"
    d1_edges_output = base_networks_path / "FTD_edges.csv"

    d1_nodes_input = base_sources_path / "UniProtKB-FTD.tsv"
    d1_nodes_output = base_networks_path / "FTD_nodes.csv"


    process_edge_file(d1_string_edge_input, d1_edges_output)
    process_node_file(d1_nodes_input, d1_nodes_output)

    # --- Esecuzione Test ---
    # Decommenta la riga sotto per eseguire la verifica automatica
    # run_tests()

if __name__ == '__main__':
    main()