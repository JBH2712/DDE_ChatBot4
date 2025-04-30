import pandas as pd
from cleanco import basename
from typing import Dict, Optional, List, Tuple
from fuzzywuzzy import fuzz



def load_dataset(path: str) -> pd.DataFrame:
    """
    Loads the company dataset.
    Supports Excel (.xlsx) and CSV (.csv).
    """
    if path.endswith(".xlsx"):
        return pd.read_excel(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError("Only .xlsx and .csv are supported.")

def preprocess_name(name: str) -> str:
    """
    Removes business suffixes, converts to lowercase, and cleans up whitespace.
    """
    if not isinstance(name, str):
        return ""
    # Lowercase
    lower = name.lower()
    # Cut out business suffixes
    base = basename(lower)
    # Remove multiple spaces and trim
    return " ".join(base.split())

def build_bvd_index(df: pd.DataFrame, id_col: str = "BvD ID number") -> Dict[int, int]:
    """
    Builds a mapping from BvD-ID → DataFrame index for O(1) lookup.
    """
    index_map: Dict[int, int] = {}
    for idx, val in df[id_col].dropna().items():
        try:
            index_map[int(val)] = idx
        except (ValueError, TypeError):
            continue
    return index_map

def fuzzy_match(
    input_name: str,
    choices: List[str]
) -> List[Tuple[str, int]]:
    """
    Compares `input_name` against each string in `choices` using
    token_set_ratio and returns a sorted list (match, score).
    """
    cleaned_input = preprocess_name(input_name)
    results: List[Tuple[str, int]] = []
    for choice in choices:
        cleaned_choice = preprocess_name(choice)
        score = fuzz.token_set_ratio(cleaned_input, cleaned_choice)
        results.append((choice, score))
    # Sort descending by score
    return sorted(results, key=lambda x: x[1], reverse=True)

def lookup_bvd_id(
    bvd_id: int,
    index_map: Dict[int, int],
    df: pd.DataFrame
) -> Optional[pd.Series]:
    """
    Looks up the company with the given BvD-ID.
    Returns the DataFrame row (Series) or None.
    """
    idx = index_map.get(int(bvd_id))
    if idx is None:
        return None
    return df.loc[idx]
