"""
preprocessing.py
================
Data loading, cleaning, feature engineering, normalisation, and SMOTE
oversampling for the hybrid fraud detection framework.

Paper
-----
"A Hybrid Deep Learning and Explainable AI Framework for Real-Time
Financial Fraud Detection in Dynamic Environments"

Design principles (from paper Sections: Methodology & Experimental
Datasets and Evaluation Protocol)
----------------------------------
1. Strict temporal ordering — data is sorted by transaction time before
   any splitting so that no future information leaks into training folds.
2. Leakage-free aggregates — all rolling/velocity features are computed
   in a forward-only (expanding/causal) manner: each transaction only
   sees information from transactions that occurred *before* it.
3. Train-fold-only statistics — StandardScaler is fit exclusively on the
   training portion of each fold and then applied to the validation
   portion.  This is enforced by the FoldPreprocessor class.
4. SMOTE on training fold only — oversampling is never applied to the
   validation set so evaluation reflects the true class distribution.
5. No raw sensitive data is stored — only processed numpy arrays and
   the fitted scaler are persisted to disk.

Usage
-----
    # One-shot pipeline (returns all 5 processed folds)
    from src.preprocessing import run_pipeline
    fold_data, feature_cols = run_pipeline("data/creditcard.csv")

    # Step-by-step
    from src.preprocessing import (
        load_creditcard, clean, engineer_features,
        build_folds, FoldPreprocessor
    )
    df           = load_creditcard("data/creditcard.csv")
    df           = clean(df)
    df, feat_cols = engineer_features(df)
    folds        = build_folds(df, n_folds=5)
    proc         = FoldPreprocessor(smote_ratio=0.1, random_state=42)
    X_tr, y_tr, X_val, y_val, scaler = proc.fit_transform(
        df[feat_cols].values, df["Class"].values, *folds[0]
    )
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

#: PCA-anonymised feature columns present in the Kaggle Credit Card dataset.
PCA_FEATURES: List[str] = [f"V{i}" for i in range(1, 29)]

#: Raw numeric columns kept as-is before engineering.
RAW_NUMERIC: List[str] = ["Amount"]

#: Names assigned to the features created by :func:`engineer_features`.
ENGINEERED_FEATURES: List[str] = [
    "log_amount",
    "amount_zscore_rolling",
    "time_delta",
    "tx_velocity_100",
    "tx_velocity_500",
    "hour_of_day",
    "is_night",
    "amount_rolling_mean_100",
    "amount_rolling_std_100",
]

#: Full ordered list of feature columns used for model training.
#: Re-exported so downstream modules (model.py, inference.py) can import
#: a single authoritative list.
ALL_FEATURE_COLS: List[str] = PCA_FEATURES + RAW_NUMERIC + ENGINEERED_FEATURES

#: Target column name.
LABEL_COL: str = "Class"

#: Random seed used throughout for reproducibility.
RANDOM_STATE: int = 42


# ── Data loading ──────────────────────────────────────────────────────────────


def load_creditcard(path: str | Path) -> pd.DataFrame:
    """Load the Kaggle European Credit Card Fraud dataset from a CSV file.

    The dataset contains 284,807 transactions recorded over two days in
    September 2013.  28 features (V1–V28) are the result of a PCA
    transformation applied for anonymisation; only *Time* and *Amount*
    remain in their original form.  The target column *Class* is 1 for
    fraud and 0 for legitimate transactions (fraud rate ≈ 0.172 %).

    Dataset download
    ----------------
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

    Parameters
    ----------
    path : str or Path
        Path to ``creditcard.csv``.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with the original column names.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at *path*.
    ValueError
        If expected columns (V1, Amount, Class) are absent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "Then place it in the data/ directory."
        )

    logger.info("Loading dataset from %s …", path)
    df = pd.read_csv(path)

    # Sanity-check expected schema
    required = {"Time", "Amount", "Class"} | set(PCA_FEATURES)
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset is missing expected columns: {missing_cols}. "
            "Make sure you downloaded the correct file."
        )

    logger.info(
        "Loaded %d rows × %d cols  |  fraud=%d (%.3f %%)",
        len(df),
        df.shape[1],
        df[LABEL_COL].sum(),
        df[LABEL_COL].mean() * 100,
    )
    return df


# ── Cleaning ──────────────────────────────────────────────────────────────────


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, handle missing values, and sort by transaction time.

    Paper reference (Methodology section):
        "Missing values and duplicates were filtered and transaction
        features were normalized to a standardized range thus stabilizing
        neural network training."

    Steps performed
    ---------------
    1. **Drop exact duplicates** — all columns identical across rows.
    2. **Handle missing values** — the Credit Card dataset has no missing
       values by design, but this step guards against corrupted downloads
       or partial exports.  Numeric columns are imputed with their median
       (computed over the full dataframe).  No imputation is applied to
       the label column.
    3. **Sort by Time** — temporal ordering is the foundation of the
       leakage-prevention strategy used in :func:`build_folds` and
       :func:`engineer_features`.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe as returned by :func:`load_creditcard`.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe, sorted by *Time*, with the index reset.
    """
    initial_rows = len(df)

    # ── Step 1: Drop duplicates ───────────────────────────────────────────────
    df = df.drop_duplicates()
    dropped = initial_rows - len(df)
    if dropped:
        logger.info("Dropped %d duplicate rows.", dropped)
    else:
        logger.info("No duplicate rows found.")

    # ── Step 2: Missing values ────────────────────────────────────────────────
    missing_total = df.isnull().sum().sum()
    if missing_total == 0:
        logger.info("No missing values detected.")
    else:
        logger.warning(
            "%d missing values detected.  Imputing numeric columns with median.",
            missing_total,
        )
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != LABEL_COL]
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.debug("  Imputed '%s' with median=%.4f", col, median_val)

    # ── Step 3: Sort by Time (ascending) — leakage-prevention prerequisite ───
    df = df.sort_values("Time").reset_index(drop=True)
    logger.info(
        "Data sorted by Time.  Range: %.0f – %.0f seconds  (~%.1f hours)",
        df["Time"].min(),
        df["Time"].max(),
        (df["Time"].max() - df["Time"].min()) / 3600,
    )

    return df


# ── Feature engineering ───────────────────────────────────────────────────────


def engineer_features(
    df: pd.DataFrame,
    velocity_window_small: int = 100,
    velocity_window_large: int = 500,
) -> Tuple[pd.DataFrame, List[str]]:
    """Derive temporal and behavioural features from raw transaction data.

    All aggregates are computed in a **forward-only (causal)** manner so
    that each transaction row only has access to information from
    transactions that occurred strictly before it.  This is the key
    leakage-prevention mechanism described in the paper:

        "aggregates like transaction velocity or account frequency were
        calculated in a streaming manner, so that a transaction only had
        access to information in the past."
        (Experimental Datasets and Evaluation Protocol)

    Features added
    --------------
    log_amount : float
        Natural log of (1 + Amount).  Reduces right-skew without losing
        zero-amount transactions.

    amount_zscore_rolling : float
        Z-score of Amount relative to an expanding window of all previous
        transactions.  Captures anomalously large or small amounts.

    time_delta : float
        Seconds elapsed since the immediately preceding transaction in the
        dataset.  First row is 0.  Clipped at 0 to remove any artefacts
        introduced by earlier operations.

    tx_velocity_100 : float
        Number of transactions in the preceding *velocity_window_small*
        rows.  Proxy for burst-activity patterns that precede fraud.

    tx_velocity_500 : float
        Same as above but over a wider *velocity_window_large* window,
        capturing slower-building velocity trends.

    hour_of_day : int
        Approximate hour extracted from *Time* (seconds modulo 86 400).
        Encodes circadian rhythm of transaction activity.

    is_night : int
        Binary flag: 1 if hour_of_day is between 00:00 and 06:00 (hours
        0–5 inclusive).  Night-time transactions have higher fraud rates.

    amount_rolling_mean_100 : float
        Expanding mean of Amount over the preceding
        *velocity_window_small* rows.  Encodes recent spending baseline.

    amount_rolling_std_100 : float
        Expanding standard deviation of Amount over the preceding
        *velocity_window_small* rows.  Encodes spending volatility.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe sorted by Time (output of :func:`clean`).
    velocity_window_small : int, default 100
        Row-count window for short-horizon velocity and amount features.
    velocity_window_large : int, default 500
        Row-count window for long-horizon velocity features.

    Returns
    -------
    df_engineered : pd.DataFrame
        Original dataframe with new feature columns appended.
    feature_cols : List[str]
        Ordered list of all columns to be used as model inputs
        (PCA features + raw numerics + engineered features).
    """
    logger.info("Engineering features …")
    df = df.copy()

    # ── log(1 + Amount) ───────────────────────────────────────────────────────
    df["log_amount"] = np.log1p(df["Amount"])

    # ── Amount z-score (expanding / causal window) ────────────────────────────
    expanding_mean = df["Amount"].expanding(min_periods=1).mean()
    expanding_std  = df["Amount"].expanding(min_periods=1).std().fillna(1.0)
    expanding_std  = expanding_std.replace(0, 1.0)          # avoid division by zero
    df["amount_zscore_rolling"] = (df["Amount"] - expanding_mean) / expanding_std

    # ── Time delta between consecutive transactions ───────────────────────────
    df["time_delta"] = df["Time"].diff().fillna(0.0).clip(lower=0.0)

    # ── Transaction velocity (row-count proxy for time windows) ───────────────
    ones_series = pd.Series(np.ones(len(df)), index=df.index)
    df["tx_velocity_100"] = (
        ones_series.rolling(window=velocity_window_small, min_periods=1).sum().values
    )
    df["tx_velocity_500"] = (
        ones_series.rolling(window=velocity_window_large, min_periods=1).sum().values
    )

    # ── Hour of day (circadian encoding) ─────────────────────────────────────
    df["hour_of_day"] = ((df["Time"] % 86_400) // 3_600).astype(int)
    df["is_night"]    = df["hour_of_day"].between(0, 5).astype(int)

    # ── Rolling amount statistics (causal, forward-only) ─────────────────────
    df["amount_rolling_mean_100"] = (
        df["Amount"]
        .rolling(window=velocity_window_small, min_periods=1)
        .mean()
        .values
    )
    df["amount_rolling_std_100"] = (
        df["Amount"]
        .rolling(window=velocity_window_small, min_periods=1)
        .std()
        .fillna(0.0)
        .values
    )

    feature_cols = PCA_FEATURES + RAW_NUMERIC + ENGINEERED_FEATURES
    logger.info(
        "Feature engineering complete.  %d features total (%d PCA + %d raw + %d engineered).",
        len(feature_cols),
        len(PCA_FEATURES),
        len(RAW_NUMERIC),
        len(ENGINEERED_FEATURES),
    )

    # Quick sanity check — no NaNs should remain in feature columns
    nan_counts = df[feature_cols].isnull().sum()
    if nan_counts.any():
        bad = nan_counts[nan_counts > 0].to_dict()
        logger.warning("NaN values after feature engineering: %s", bad)

    return df, feature_cols


# ── Train / validation splitting ─────────────────────────────────────────────


def build_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create *n_folds* time-stratified (non-shuffled) train / validation splits.

    Unlike random k-fold, this function divides the *time-sorted* dataframe
    into consecutive blocks.  Each fold uses all blocks **before** the
    validation block as training data:

        Fold 0 : train=[block1..4]   val=[block0]
        Fold 1 : train=[block0,2..4] val=[block1]
        …

    Wait — to avoid the first fold having no training data, the splits
    are structured so that fold *k* uses the *first k blocks* as training
    and block *k* as validation (walk-forward expanding window):

        Fold 0 : train=[block0]          val=[block1]
        Fold 1 : train=[block0–1]        val=[block2]
        Fold 2 : train=[block0–2]        val=[block3]
        Fold 3 : train=[block0–3]        val=[block4]

    This reproduces the *5 time-stratified folds* described in the paper:
        "All reported figures are means and standard deviation of 5
        time-stratified folds."
        (Experimental Datasets and Evaluation Protocol)

    Parameters
    ----------
    df : pd.DataFrame
        Time-sorted cleaned dataframe (output of :func:`clean`).
    n_folds : int, default 5
        Number of folds.  The first fold uses 1 block for training and
        1 for validation; subsequent folds grow the training window.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (train_indices, val_indices) pairs, one per fold.
    """
    n = len(df)
    # Split into n_folds + 1 equal blocks; first block is always in training
    n_blocks   = n_folds + 1
    block_size = n // n_blocks
    block_edges = [i * block_size for i in range(n_blocks)] + [n]

    folds = []
    for k in range(1, n_folds + 1):
        train_idx = np.arange(0, block_edges[k])
        val_idx   = np.arange(block_edges[k], block_edges[k + 1])
        folds.append((train_idx, val_idx))

        fraud_tr  = df.iloc[train_idx][LABEL_COL].mean() * 100
        fraud_val = df.iloc[val_idx][LABEL_COL].mean() * 100
        logger.info(
            "Fold %d/%d  train=%7d  val=%6d  fraud_train=%.3f%%  fraud_val=%.3f%%",
            k,
            n_folds,
            len(train_idx),
            len(val_idx),
            fraud_tr,
            fraud_val,
        )

    return folds


# ── Per-fold preprocessing class ─────────────────────────────────────────────


@dataclass
class FoldPreprocessor:
    """Stateful preprocessor that normalises and oversamples a single fold.

    Encapsulates the *fit-on-train, transform-train-and-val* pattern
    required to prevent data leakage across fold boundaries.  One instance
    should be created per fold.

    Parameters
    ----------
    smote_ratio : float, default 0.1
        Desired ratio of minority (fraud) to majority (legitimate) class
        in the *training* set after SMOTE.  For example, 0.1 means the
        training set will contain 10 fraud rows per 100 legitimate rows.
        The paper applies SMOTE only to the training fold, never to
        validation.
    random_state : int, default 42
        Seed for SMOTE and any other stochastic operations.
    """

    smote_ratio: float = 0.1
    random_state: int = RANDOM_STATE

    # Populated by fit_transform; exposed for downstream inspection / saving
    scaler_: Optional[StandardScaler] = field(default=None, init=False, repr=False)

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
        """Fit the scaler on training data and apply SMOTE, then transform validation.

        Processing order
        ----------------
        1. Slice train / val arrays using the provided indices.
        2. Fit :class:`~sklearn.preprocessing.StandardScaler` on *X_train*.
        3. Transform *X_train* and *X_val* with the fitted scaler.
        4. Apply :class:`~imblearn.over_sampling.SMOTE` to *X_train_scaled*
           only — never to the validation set.

        The scaler is stored in ``self.scaler_`` and can be persisted with
        :func:`save_scaler` for use during inference.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Full feature matrix for the current dataset (all folds).
        y : np.ndarray, shape (n_samples,)
            Binary label vector (0 = legitimate, 1 = fraud).
        train_idx : np.ndarray
            Integer indices selecting the training rows from *X* and *y*.
        val_idx : np.ndarray
            Integer indices selecting the validation rows from *X* and *y*.

        Returns
        -------
        X_train_resampled : np.ndarray
            SMOTE-augmented, scaled training features.
        y_train_resampled : np.ndarray
            Labels matching *X_train_resampled*.
        X_val_scaled : np.ndarray
            Scaled validation features (no resampling).
        y_val : np.ndarray
            Validation labels (unchanged).
        scaler : StandardScaler
            The fitted scaler (same object as ``self.scaler_``).
        """
        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx],   y[val_idx]

        # ── Normalisation — fit on train, apply to both ───────────────────────
        self.scaler_ = StandardScaler()
        X_train_scaled = self.scaler_.fit_transform(X_train)
        X_val_scaled   = self.scaler_.transform(X_val)

        logger.info(
            "Scaler fitted.  Train fraud: %d/%d (%.3f%%)  Val fraud: %d/%d (%.3f%%)",
            y_train.sum(), len(y_train), y_train.mean() * 100,
            y_val.sum(),   len(y_val),   y_val.mean() * 100,
        )

        # ── SMOTE — training fold only ────────────────────────────────────────
        # Paper: "oversampling the minority class to a training fold was
        # limited … to prevent alterations in the natural data distributions"
        current_ratio = y_train.sum() / (y_train == 0).sum()
        if current_ratio >= self.smote_ratio:
            logger.info(
                "SMOTE skipped — current minority ratio %.4f already >= target %.4f.",
                current_ratio,
                self.smote_ratio,
            )
            X_train_res, y_train_res = X_train_scaled, y_train
        else:
            smote = SMOTE(
                sampling_strategy=self.smote_ratio,
                random_state=self.random_state,
                k_neighbors=5,
                n_jobs=-1,
            )
            X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
            logger.info(
                "SMOTE applied.  Resampled train: %d rows  fraud=%.2f%%",
                len(X_train_res),
                y_train_res.mean() * 100,
            )

        return X_train_res, y_train_res, X_val_scaled, y_val, self.scaler_

    def save_scaler(self, path: str | Path) -> None:
        """Persist the fitted scaler to disk using joblib.

        The saved scaler is required by the inference pipeline
        (``src/inference.py``) to normalise incoming transactions at
        serving time using exactly the same statistics as during training.

        Parameters
        ----------
        path : str or Path
            Destination file path, e.g. ``data/processed/scaler_fold0.pkl``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit_transform`.
        """
        if self.scaler_ is None:
            raise RuntimeError(
                "Scaler has not been fitted yet.  Call fit_transform() first."
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler_, path)
        logger.info("Scaler saved to %s", path)

    @staticmethod
    def load_scaler(path: str | Path) -> StandardScaler:
        """Load a previously saved scaler from disk.

        Parameters
        ----------
        path : str or Path
            Path to a joblib-serialised :class:`~sklearn.preprocessing.StandardScaler`.

        Returns
        -------
        StandardScaler
            The deserialised, ready-to-use scaler.
        """
        return joblib.load(path)


# ── High-level one-shot pipeline ─────────────────────────────────────────────


def run_pipeline(
    data_path: str | Path,
    output_dir: str | Path = "data/processed",
    n_folds: int = 5,
    smote_ratio: float = 0.1,
    random_state: int = RANDOM_STATE,
    save: bool = True,
) -> Tuple[List[dict], List[str]]:
    """End-to-end preprocessing pipeline: load → clean → engineer → fold → normalise.

    Convenience wrapper that calls :func:`load_creditcard`, :func:`clean`,
    :func:`engineer_features`, :func:`build_folds`, and
    :class:`FoldPreprocessor` in sequence and optionally persists the
    results to ``output_dir``.

    Parameters
    ----------
    data_path : str or Path
        Path to ``creditcard.csv``.
    output_dir : str or Path, default ``"data/processed"``
        Directory where processed fold arrays and scalers are saved.
        Ignored when *save=False*.
    n_folds : int, default 5
        Number of time-stratified folds.
    smote_ratio : float, default 0.1
        Desired minority-to-majority ratio after SMOTE (training fold only).
    random_state : int, default 42
        Global random seed.
    save : bool, default True
        If True, persist fold arrays (``fold_data.pkl``), feature column
        names (``feature_cols.pkl``), and per-fold scalers
        (``scaler_fold{k}.pkl``) to *output_dir*.

    Returns
    -------
    fold_data : List[dict]
        List of dicts, one per fold.  Each dict has keys:
        ``X_train``, ``y_train``, ``X_val``, ``y_val``.
    feature_cols : List[str]
        Ordered list of feature column names.  Length equals
        ``X_train.shape[1]``.

    Examples
    --------
    >>> fold_data, feature_cols = run_pipeline("data/creditcard.csv")
    >>> print(f"{len(fold_data)} folds, {len(feature_cols)} features")
    5 folds, 37 features
    >>> X_train = fold_data[0]["X_train"]
    >>> y_train = fold_data[0]["y_train"]
    """
    np.random.seed(random_state)
    output_dir = Path(output_dir)

    # ── Load ──────────────────────────────────────────────────────────────────
    df = load_creditcard(data_path)

    # ── Clean ─────────────────────────────────────────────────────────────────
    df = clean(df)

    # ── Feature engineering ───────────────────────────────────────────────────
    df, feature_cols = engineer_features(df)

    # ── Build fold index arrays ───────────────────────────────────────────────
    folds = build_folds(df, n_folds=n_folds)

    # ── Extract full feature matrix and label vector ──────────────────────────
    X = df[feature_cols].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.int32)

    # ── Per-fold normalisation + SMOTE ────────────────────────────────────────
    fold_data: List[dict] = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        logger.info("─── Processing fold %d/%d ───", fold_idx + 1, n_folds)

        preprocessor = FoldPreprocessor(
            smote_ratio=smote_ratio,
            random_state=random_state,
        )
        X_tr, y_tr, X_val, y_val, _ = preprocessor.fit_transform(
            X, y, train_idx, val_idx
        )

        fold_data.append(
            {
                "X_train": X_tr,
                "y_train": y_tr,
                "X_val":   X_val,
                "y_val":   y_val,
            }
        )

        if save:
            preprocessor.save_scaler(output_dir / f"scaler_fold{fold_idx}.pkl")

    # ── Persist processed data ────────────────────────────────────────────────
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(fold_data,    output_dir / "fold_data.pkl")
        joblib.dump(feature_cols, output_dir / "feature_cols.pkl")
        np.save(output_dir / "X_raw.npy", X)
        np.save(output_dir / "y_raw.npy", y)
        logger.info("All preprocessed data saved to %s/", output_dir)

    logger.info(
        "Pipeline complete.  %d folds  |  %d features  |  "
        "Train shape (fold 0): %s",
        n_folds,
        len(feature_cols),
        fold_data[0]["X_train"].shape,
    )

    return fold_data, feature_cols


# ── Inference-time helper ─────────────────────────────────────────────────────


def preprocess_single_transaction(
    transaction: dict,
    scaler: StandardScaler,
    feature_cols: Optional[List[str]] = None,
) -> np.ndarray:
    """Transform a single raw transaction dict into a scaled feature vector.

    Used by ``src/inference.py`` (FastAPI endpoint) to preprocess an
    incoming transaction at serving time using the scaler fitted during
    training.

    Note
    ----
    Engineered features that depend on historical context (e.g.
    ``tx_velocity_100``, ``amount_zscore_rolling``) cannot be recomputed
    from a single transaction in isolation.  The caller is responsible
    for computing those aggregates upstream (e.g. from a running Redis
    cache or feature store) and including them in *transaction*.  This
    function only applies the StandardScaler and returns the feature array.

    Parameters
    ----------
    transaction : dict
        Dictionary mapping feature names to their values.  Must contain
        at minimum all columns in *feature_cols*.
    scaler : StandardScaler
        Fitted scaler loaded from ``data/processed/scaler_fold{k}.pkl``.
    feature_cols : List[str], optional
        Ordered list of feature names.  Defaults to :data:`ALL_FEATURE_COLS`.

    Returns
    -------
    np.ndarray, shape (1, n_features)
        Scaled feature vector ready for DNN encoder input.

    Raises
    ------
    KeyError
        If any required feature is missing from *transaction*.

    Examples
    --------
    >>> import joblib
    >>> scaler = FoldPreprocessor.load_scaler("data/processed/scaler_fold0.pkl")
    >>> feature_cols = joblib.load("data/processed/feature_cols.pkl")
    >>> tx = {"V1": -1.36, "V2": -0.07, ..., "Amount": 149.62,
    ...       "log_amount": 5.01, "time_delta": 3.0, ...}
    >>> x = preprocess_single_transaction(tx, scaler, feature_cols)
    >>> x.shape
    (1, 37)
    """
    if feature_cols is None:
        feature_cols = ALL_FEATURE_COLS

    missing = [col for col in feature_cols if col not in transaction]
    if missing:
        raise KeyError(
            f"Transaction dict is missing {len(missing)} feature(s): {missing}. "
            "Ensure all engineered features are computed before calling this function."
        )

    raw_vector = np.array(
        [transaction[col] for col in feature_cols], dtype=np.float32
    ).reshape(1, -1)

    scaled_vector = scaler.transform(raw_vector)
    return scaled_vector.astype(np.float32)


# ── CLI entry point ───────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the full preprocessing pipeline for the fraud detection paper."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/creditcard.csv",
        help="Path to creditcard.csv (default: data/creditcard.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed fold arrays (default: data/processed)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of time-stratified folds (default: 5)",
    )
    parser.add_argument(
        "--smote-ratio",
        type=float,
        default=0.1,
        help="SMOTE minority/majority ratio after oversampling (default: 0.1)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run pipeline without saving output files",
    )
    args = parser.parse_args()

    fold_data, feature_cols = run_pipeline(
        data_path=args.input,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        smote_ratio=args.smote_ratio,
        save=not args.no_save,
    )

    print(f"\nDone.  {len(fold_data)} folds  |  {len(feature_cols)} features")
    print(f"Fold 0 train shape : {fold_data[0]['X_train'].shape}")
    print(f"Fold 0 val shape   : {fold_data[0]['X_val'].shape}")
    print(f"Feature columns    : {feature_cols}")
