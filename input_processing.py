import pandas as pd
from typing import List, Optional, Tuple
import streamlit as st

from data_preprocessing import preprocess_name, fuzzy_match, build_bvd_index, lookup_bvd_id

# Threshold for unique matching
DEFAULT_MATCH_THRESHOLD = 80  

def get_user_input() -> Tuple[str, str, Optional[int]]:
    """
    Reads the user input in Streamlit:
      - company_name (TextInput)
      - company_description (TextArea, optional)
      - bvd_id (NumberInput, optional)
    """
    name = st.text_input("Company name")
    desc = st.text_area("Description (optional)", height=100)
    bvd_id = st.number_input(
        "BvD ID number (optional)", 
        min_value=0, step=1, format="%d"
    )
    # If 0, treat as None
    bvd_id = None if bvd_id == 0 else int(bvd_id)
    return name.strip(), desc.strip(), bvd_id

def match_company(
    user_name: str,
    user_desc: str,
    user_bvd: Optional[int],
    df: pd.DataFrame,
    bvd_index: dict,
    threshold: int = DEFAULT_MATCH_THRESHOLD
) -> Tuple[Optional[pd.Series], List[Tuple[str,int]]]:
    """
    Tries to uniquely match the company.
    1. If user_bvd is set → Direct lookup via build_bvd_index
    2. Otherwise: fuzzy_match using the name
       - Score ≥ threshold → unique match
       - Otherwise return Top-5 suggestions
    Returns:
      - matched_row (pd.Series) or None
      - suggestions: List[(company_name, score)] (max. 5 entries)
    """
    # 1) BvD-ID lookup
    if user_bvd is not None:
        row = lookup_bvd_id(user_bvd, bvd_index, df)
        if row is not None:
            return row, []
        # if invalid: continue with fuzzy matching

    # 2) Fuzzy Name Matching
    all_names: List[str] = df["Company name Latin alphabet"].tolist()
    matches = fuzzy_match(user_name, all_names)
    
    # unique match?
    top_name, top_score = matches[0]
    if top_score >= threshold:
        # return the matched row
        matched_row = df[df["Company name Latin alphabet"] == top_name].iloc[0]
        return matched_row, []
    
    # otherwise Top-5 suggestions
    suggestions = matches[:5]
    return None, suggestions
