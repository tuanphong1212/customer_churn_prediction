import numpy as np
import pandas as pd

def calculate_psi_continuous(reference : pd.Series , current : pd.Series , buckets : int = 10) -> float:
    """PSI cho feature dạng số (liên tục) - chia bucket theo quatile của reference"""
    breakpoints = np.quantile(reference , np.linspace(0 , 1 , buckets + 1))
    breakpoints  = np.unique(breakpoints)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_pct = pd.cut(reference , bins=breakpoints).value_counts(normalize=True , sort=False)
    cur_pct = pd.cut(current , bins=breakpoints).value_counts(normalize=True , sort=False)
    cur_pct = cur_pct.reindex(ref_pct.index , fill_value=0)

    ref_pct = ref_pct.replace(0 , 1e-4)
    cur_pct = cur_pct.replace(0 , 1e-4)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

def calculate_psi_categorical(reference : pd.Series , current : pd.Series) -> float:
    """PSI cho feature dạng phân loại - dùng tỉ lệ category thay cho buckets"""
    ref_pct = reference.value_counts(normalize=True)
    cur_pct = current.value_counts(normalize=True).reindex(ref_pct.index , fill_value=0)

    ref_pct = ref_pct.replace(0 , 1e-4)
    cur_pct = cur_pct.replace(0 , 1e-4)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

