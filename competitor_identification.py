# competitor_identification.py

import pandas as pd
from typing import List, Set, Dict, Any

def get_industry_set(row: pd.Series, industry_cols: List[str]) -> Set[str]:
    """
    Reads all non-empty industry fields from a DataFrame row
    and returns them as a set.
    """
    return {
        row[col] for col in industry_cols
        if pd.notna(row.get(col)) and str(row.get(col)).strip() != ""
    }

def industry_pair_count(
    input_row: pd.Series,
    candidate_row: pd.Series,
    industry_cols: List[str]
) -> int:
    """
    Number of matching industries between input and candidate.
    """
    s1 = get_industry_set(input_row, industry_cols)
    s2 = get_industry_set(candidate_row, industry_cols)
    return len(s1 & s2)

def industry_overlap(
    input_row: pd.Series,
    candidate_row: pd.Series,
    industry_cols: List[str]
) -> float:
    """
    Percentage of the input_row industries that also
    appear in candidate_row (0–100).
    """
    s1 = get_industry_set(input_row, industry_cols)
    if not s1:
        return 0.0
    s2 = get_industry_set(candidate_row, industry_cols)
    return (len(s1 & s2) / len(s1)) * 100.0

def nace_exact_match(
    input_row: pd.Series,
    candidate_row: pd.Series,
    nace_col: str
) -> bool:
    """
    Checks whether the NACE code in input_row and candidate_row are the same.
    """
    return pd.notna(input_row.get(nace_col)) \
       and input_row.get(nace_col) == candidate_row.get(nace_col)

def evaluate_candidate(
    input_row: pd.Series,
    candidate_row: pd.Series,
    industry_cols: List[str],
    nace_col: str
) -> Dict[str, Any]:
    """
    Determines all three criteria and counts how many are met.
    Returns a dict with:
      - pair_count (int)
      - overlap      (float)
      - nace_match   (bool)
      - num_criteria (int)
    """
    pair = industry_pair_count(input_row, candidate_row, industry_cols)
    overlap = industry_overlap(input_row, candidate_row, industry_cols)
    nace_ok = nace_exact_match(input_row, candidate_row, nace_col)
    
    # Criterion 1: overlap >= 50%
    c1 = overlap >= 50.0
    # Criterion 2: pair_count >= 2
    c2 = pair >= 2
    # Criterion 3: exact NACE match
    c3 = nace_ok
    
    return {
        "pair_count": pair,
        "overlap": overlap,
        "nace_match": nace_ok,
        "num_criteria": sum([c1, c2, c3])
    }

def find_potential_competitors(
    input_row: pd.Series,
    df: pd.DataFrame,
    industry_cols: List[str],
    nace_col: str
) -> pd.DataFrame:
    """
    Determines from df all candidates that meet at least one of the criteria,
    groups them by the number of criteria met, and sorts descending
    by (num_criteria, pair_count).
    
    Returns: DataFrame with columns
      ['Company name Latin alphabet', 'pair_count', 'overlap', 'nace_match', 'num_criteria']
    """
    records = []
    for idx, row in df.iterrows():
        # Exclude the input company
        if idx == input_row.name:
            continue
        metrics = evaluate_candidate(input_row, row, industry_cols, nace_col)
        if metrics["num_criteria"] >= 1:
            rec = {
                "Company name Latin alphabet": row["Company name Latin alphabet"],
                **metrics
            }
            records.append(rec)
    
    result = pd.DataFrame(records)
    # Sort first by number of criteria met, then by industry pair count
    if not result.empty:
        result = result.sort_values(
            by=["num_criteria", "pair_count"], ascending=False
        ).reset_index(drop=True)
    return result
