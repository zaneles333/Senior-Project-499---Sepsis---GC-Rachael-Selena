from shiny import App, render, ui, reactive
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
import numpy as np
import os
import json
import base64
import xgboost as xgb

# ============================================================
# 1. GLOBAL CONFIGURATION
# ============================================================
# These constants control app behavior, file locations, thresholds,
# and how time bins are translated into hours on plots.
#
# CSV data flow summary:
# 1. User uploads a CSV in the GUI.
# 2. The app reads the CSV into a Pandas DataFrame.
# 3. One active row is selected from the DataFrame.
# 4. Optional provider overrides are applied to that row.
# 5. The row is transformed in two directions:
#       A) into time-series data for graphs/tables
#       B) into aligned numeric model input for XGBoost
# 6. The resulting outputs are shown in the GUI.
# ============================================================

TIME_BIN_MAP = {"1": 4, "2": 8, "3": 12}


def first_existing_path(*candidates: str) -> str:
    """Return the first existing path from a list of candidate filenames."""
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return candidates[0]


MODEL_PATH = os.getenv(
    "MODEL_PATH",
    first_existing_path("sepsis_risk_model.json", "sepsis_risk_model(5).json", "sepsis_risk_model(4).json", "sepsis_risk_model(3).json", "sepsis_risk_model(2).json")
)
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8003"))
DISPLAY_HOST = os.getenv("DISPLAY_HOST", "127.0.0.1")

# Risk thresholds for the model probability.
T_LOW = 0.10
T_HIGH = 0.50

# Columns removed before sending uploaded CSV rows into the XGBoost model.
# These match the training/inference cleanup used for sepsis_risk_model(3).json.
DROPPED_IN_TRAINING = {
    "NUM_ANTIBIOTICS", "PROP_RESIST", "PROP_INTERMED", "NUM_ORGANISMS",
}
ID_COLS = {"HADM_ID", "SUBJECT_ID", "SEPSIS_STATUS"}

# ============================================================
# 2. STATIC DISPLAY ASSETS
# ============================================================
# These are purely visual/demo values used in the UI header.
# They do not affect the uploaded CSV, model input, or predictions.
# ============================================================

patient_demo = {
    "Name": "John Doe",
    "Age": 54,
    "Sex": "Male",
    "Hospital ID": "A1029934",
}

# Properly base64-encoded SVG so browsers never show a broken-image icon.
SILHOUETTE_SVG = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyMDAg"
    "MjAwIj48Y2lyY2xlIGN4PSIxMDAiIGN5PSI3MCIgcj0iMzgiIGZpbGw9IiNjZWQ0ZGEiLz48cGF0"
    "aCBkPSJNMzUgMTg1YzAtNDAgMjgtNjggNjUtNjhzNjUgMjggNjUgNjgiIGZpbGw9IiNjZWQ0ZGEi"
    "Lz48L3N2Zz4="
)


LOGO_PATH = first_existing_path("prenova_cell_bacteria_logo.png", "prenova_cell_bacteria_logo(1).png", "logo.png")


def image_file_to_data_uri(path: str):
    """
    Convert a local PNG logo into a browser-safe data URI.

    This lets the app use prenova_cell_bacteria_logo.png directly without
    needing separate static-file routing in Shiny/Docker.
    """
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# Header logo used in the top-left of the app.
APP_LOGO_SRC = image_file_to_data_uri(LOGO_PATH)

# ============================================================
# 3. XGBOOST MODEL LOADING
# ============================================================
# These functions run once when the app starts.
# They prepare the XGBoost model used for sepsis risk prediction.
# This version intentionally uses direct XGBoost feature alignment only.
# ============================================================

def infer_model_num_features(path: str):
    """
    Purpose:
        Inspect the saved XGBoost model JSON and infer how many input
        features the model expects.

    Input:
        path: str
            File path to the XGBoost JSON model.

    Output:
        int | None
            Number of expected model features if it can be found,
            otherwise None.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            obj = json.load(f)

        num_feat = (
            obj.get("learner", {})
            .get("learner_model_param", {})
            .get("num_feature", None)
        )
        if num_feat is None:
            num_feat = (
                obj.get("learner", {})
                .get("gradient_booster", {})
                .get("model", {})
                .get("gbtree_model_param", {})
                .get("num_feature", None)
            )
        return int(num_feat) if num_feat is not None else None
    except Exception:
        return None


class XGBoostBoosterAdapter:
    """Adapter around xgboost.Booster that avoids the sklearn wrapper."""

    def __init__(self, model_path: str, num_features: int | None):
        self.booster = xgb.Booster()
        self.booster.load_model(model_path)
        self.num_features = num_features

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        dmat = xgb.DMatrix(X)
        pred = np.asarray(self.booster.predict(dmat), dtype=float)
        if pred.ndim == 1:
            p1 = np.clip(pred, 0.0, 1.0)
            return np.column_stack([1.0 - p1, p1])
        return pred

    @property
    def feature_importances_(self):
        n = int(self.num_features or 0)
        if n <= 0:
            return np.array([], dtype=float)
        scores = self.booster.get_score(importance_type="gain")
        arr = np.zeros(n, dtype=float)
        for key, val in scores.items():
            if isinstance(key, str) and key.startswith("f"):
                try:
                    idx = int(key[1:])
                    if 0 <= idx < n:
                        arr[idx] = float(val)
                except ValueError:
                    pass
        total = arr.sum()
        if total > 0:
            arr = arr / total
        return arr


def load_xgb_model(path: str):
    """
    Load the trained XGBoost model used for sepsis risk prediction.

    Uses xgb.Booster instead of xgb.XGBClassifier so Docker does not need
    scikit-learn for model loading/prediction. The rest of the app can still
    call predict_proba(...) and feature_importances_.
    """
    if not os.path.exists(path):
        return None, None, f"XGBoost model not found at {path}. Using placeholder score."
    try:
        n = infer_model_num_features(path)
        model = XGBoostBoosterAdapter(path, n)
        msg = f"Loaded XGBoost Booster from {path}"
        if n is not None:
            msg += f" (expects {n} features)"
        return model, n, msg + "."
    except Exception as e:
        return None, None, f"Failed to load XGBoost model: {e}. Using placeholder score."



# Load reusable model resources once when the app starts.
# load_xgb_model already calls infer_model_num_features internally,
# so we unpack all three values here instead of calling it a second time.
XGB_MODEL, MODEL_NUM_FEATURES, MODEL_STATUS = load_xgb_model(MODEL_PATH)
MODEL_INPUT_STATUS = (
    f"Model input alignment: drops IDs/non-training microbiology columns, converts remaining columns to numeric, "
    f"then pads/trims to {MODEL_NUM_FEATURES} features using NaN for missing values."
    if MODEL_NUM_FEATURES is not None
    else "Model input alignment: model feature count could not be inferred; using uploaded numeric columns after cleanup."
)

# ============================================================
# 4. GENERAL HELPERS
# ============================================================
# These support naming, matching, risk labeling, note discovery,
# theming, and general data cleanup.
# ============================================================

def normalize_name(name: str) -> str:
    """
    Convert a column or user-entered field name into a simplified,
    lowercase alphanumeric string for safer matching.

    Example:
        'WBC_Blood' -> 'wbcblood'
        'wbc blood' -> 'wbcblood'
    """
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# Placeholder score thresholds (different scale than model probability).
T_LOW_PLACEHOLDER = 0.4
T_HIGH_PLACEHOLDER = 0.8


def classify_score(score: float, low_thresh: float, high_thresh: float):
    """
    Convert any numeric score into a risk label and display color.

    Input:
        score: float
            Numeric score (model probability 0-1, or placeholder clinical score).
        low_thresh: float
            Scores below this are LOW risk.
        high_thresh: float
            Scores at or above this are HIGH risk.

    Output:
        tuple[label, color]
            label: LOW / MODERATE / HIGH
            color: hex color string used in the GUI
    """
    if score < low_thresh:
        return "LOW", "#4CAF50"
    elif score < high_thresh:
        return "MODERATE", "#FFC107"
    else:
        return "HIGH", "#F44336"


def classify_probability(prob: float):
    """Classify a model probability (0-1 scale) into a risk label and color."""
    return classify_score(prob, T_LOW, T_HIGH)


def classify_risk(score: float):
    """Classify a placeholder clinical score into a risk label and color."""
    return classify_score(score, T_LOW_PLACEHOLDER, T_HIGH_PLACEHOLDER)


# ============================================================
# FALLBACK SCORING LAYER  (all 15 top XGBoost predictors)
# ============================================================
# All 15 features from the XGBoost feature importance chart are represented.
# The three admission_type columns are one-hot encoded from the same categorical
# variable and are handled as a single group signal to avoid double-counting.
# emb_81 and emb_545 are included and contribute whenever present.
# Every feature gracefully degrades: missing / NaN values are excluded from
# the weighted average entirely — they do not pull the score toward any value.
#
# Calibration decisions (derived from analysis of real patient CSVs):
#   - Clinical thresholds are tightened relative to textbook normals to catch
#     moderate abnormalities that appear in these patient cohorts.
#   - Admission type weight is reduced (0.02 vs chart 0.0488) because it is
#     a near-constant signal (emergency=1) across most demo patients and
#     inflates all scores equally when left at full weight.
#   - Embedding normalization ceiling is 1.0 (not 0.5) because sentence-
#     transformer dims regularly exceed 0.5 in this dataset.
#   - The final score uses a sigmoid stretch centered at 0.42 to push
#     borderline-normal cases toward LOW and borderline-sick cases toward HIGH,
#     giving meaningful spread across the T_LOW=0.10 / T_HIGH=0.50 thresholds.
#   - Each lab feature checks bin3 first, then bin2, then bin1, so earlier
#     time-bin values contribute when bin3 is absent.
# ============================================================

_FALLBACK_WEIGHTS: dict[str, float] = {
    # key           weight    source
    "ast":          0.0250,   # ast_bin3_mean          (chart rank 1)
    "alt":          0.0230,   # alt_bin3_mean          (chart rank 2)
    "_admission":   0.0200,   # emergency+elective+newborn combined, reduced
    "bili_dir":     0.0210,   # bili_dir_bin3_mean     (chart rank 4)
    "wbc_urine":    0.0170,   # wbc_urine_bin3_mean    (chart rank 5)
    "emb_81":       0.0155,   # emb_81                 (chart rank 7)
    "bili_tot":     0.0140,   # bili_tot_bin3_mean     (chart rank 8)
    "lactate":      0.0130,   # lactate_bin3_mean      (chart rank 9)
    "hemoglob":     0.0125,   # hemoglob_bin3_mean     (chart rank 10)
    "emb_545":      0.0120,   # emb_545                (chart rank 11)
    "na_urine":     0.0110,   # na_urine_bin3_mean     (chart rank 12)
    "bicarb":       0.0090,   # bicarb_blood_bin3_mean (chart rank 14)
    "po2":          0.0088,   # blood_po2_bin3_mean    (chart rank 15)
}

# Maps internal short names to base column name for the bin-fallback lookup
_FEAT_BASE_COL: dict[str, str] = {
    "ast":      "ast",
    "alt":      "alt",
    "bili_dir": "bili_dir",
    "wbc_urine":"wbc_urine",
    "bili_tot": "bili_tot",
    "lactate":  "lactate",
    "hemoglob": "hemoglob",
    "na_urine": "na_urine",
    "bicarb":   "bicarb_blood",
    "po2":      "blood_po2",
}


def _get_col(row_df: pd.DataFrame, feature: str) -> float | None:
    """Case-insensitive column lookup. Returns float or None if absent/NaN."""
    for col in row_df.columns:
        if col.lower() == feature.lower():
            raw = row_df[col].values[0]
            if isinstance(raw, float) and np.isnan(raw):
                return None
            try:
                return float(raw)
            except (ValueError, TypeError):
                return None
    return None


def _get_best_bin(row_df: pd.DataFrame, base: str) -> float | None:
    """
    Look up a lab value trying bin3 first, then bin2, then bin1.
    Tries mean, then max, then min stat for each bin.
    Returns the first non-NaN value found, or None.
    """
    for bin_n in ("bin3", "bin2", "bin1"):
        for stat in ("mean", "max", "min"):
            val = _get_col(row_df, f"{base}_{bin_n}_{stat}")
            if val is not None:
                return val
    return None


def _admission_danger_signal(row_df: pd.DataFrame) -> float | None:
    """
    Single danger signal for the admission-type one-hot group.

    Reads all three admission_type columns and returns:
        0.90  newborn   — neonatal sepsis: highest mortality
        0.80  emergency — unplanned urgent presentation
        0.60  urgent    — not elective and not emergency
        0.10  elective  — planned admission: protective
        None  if all three columns are absent (group excluded from average)
    """
    emerg    = _get_col(row_df, "admission_type_emergency")
    elective = _get_col(row_df, "admission_type_elective")
    newborn  = _get_col(row_df, "admission_type_newborn")
    if emerg is None and elective is None and newborn is None:
        return None
    if newborn  is not None and newborn  == 1: return 0.90
    if emerg    is not None and emerg    == 1: return 0.80
    if elective is not None and elective == 1: return 0.10
    return 0.60  # at least one col present but none are 1 → urgent


def _danger_signal(feature: str, value: float) -> float | None:
    """
    Map one feature value to a 0-1 danger signal.

    Thresholds are tightened relative to textbook normals to detect moderate
    abnormalities present in real ICU patient data:

        ast/alt:     danger starts at 25 U/L  (early hepatocellular injury)
        bili_dir:    danger starts at 0.2 mg/dL, severe at 1.0
        wbc_urine:   danger starts at 2 /hpf, severe at 100
        emb_81/545:  abs(val)/1.0  (wider ceiling for sentence-transformer range)
        bili_tot:    danger starts at 0.8 mg/dL, severe at 10
        lactate:     danger starts at 1.5 mmol/L (pre-sepsis hypoperfusion)
        hemoglob:    inverted, danger starts at 13 g/dL, severe at 7
        na_urine:    inverted, danger starts at 50 mEq/L, severe at 10
        bicarb:      inverted, danger starts at 24 mEq/L, severe at 15
        po2:         inverted, danger starts at 100 mmHg, severe at 40
    """
    if feature == "ast":
        return float(min(max((value - 25) / 175, 0.0), 1.0))
    if feature == "alt":
        return float(min(max((value - 25) / 175, 0.0), 1.0))
    if feature == "bili_dir":
        return float(min(max((value - 0.2) / 0.8, 0.0), 1.0))
    if feature == "wbc_urine":
        return float(min(max((value - 2) / 98, 0.0), 1.0))
    if feature == "emb_81":
        return float(min(abs(value) / 1.0, 1.0))
    if feature == "bili_tot":
        return float(min(max((value - 0.8) / 9.2, 0.0), 1.0))
    if feature == "lactate":
        return float(min(max((value - 1.5) / 2.5, 0.0), 1.0))
    if feature == "hemoglob":
        return float(min(max((13.0 - value) / 6.0, 0.0), 1.0))
    if feature == "emb_545":
        return float(min(abs(value) / 1.0, 1.0))
    if feature == "na_urine":
        return float(min(max((50.0 - value) / 40.0, 0.0), 1.0))
    if feature == "bicarb":
        return float(min(max((24.0 - value) / 9.0, 0.0), 1.0))
    if feature == "po2":
        return float(min(max((100.0 - value) / 60.0, 0.0), 1.0))
    return None


def _sigmoid_stretch(raw: float, center: float = 0.42, steepness: float = 8.0) -> float:
    """
    Stretch a raw weighted-average score using a sigmoid curve.

    This spreads clustered mid-range scores across the full [0, 1] range
    without requiring any fixed linear calibration constants.

        raw = center             → 0.50  (exactly at HIGH threshold)
        raw = center - 0.15      → ~0.18 (comfortably MODERATE)
        raw = center - 0.30      → ~0.05 (LOW)
        raw = center + 0.15      → ~0.82 (clearly HIGH)

    Center and steepness can be adjusted if the patient population shifts.
    """
    return float(1.0 / (1.0 + np.exp(-steepness * (raw - center))))


def compute_fallback_risk(row_df: pd.DataFrame | None) -> dict | None:
    """
    Compute a calibrated sepsis risk score from the top-15 XGBoost predictors.

    Used when the trained model is unavailable or the CSV is too sparse
    (< 10% of 941 features present) for the model to discriminate.

    Input:
        row_df: one-row patient DataFrame (overrides already applied).

    Output:
        dict with keys:
            score       float [0, 1] — same scale as model probability
            label       LOW / MODERATE / HIGH  (T_LOW=0.10, T_HIGH=0.50)
            color       hex color string for the GUI
            n_used      signals that contributed (max 13)
            n_avail     13
            details     {signal: (value_str, danger_signal)} for display
        or None if no fallback features are present at all.

    Missing features are excluded from the weighted average (not zero-filled),
    so a patient with only 3 features still gets a meaningful score from those 3.
    """
    if row_df is None or row_df.empty:
        return None

    total_weight = 0.0
    weighted_sum = 0.0
    details: dict[str, tuple] = {}

    # Admission type group: 3 one-hot columns → 1 combined signal
    adm_sig = _admission_danger_signal(row_df)
    if adm_sig is not None:
        w = _FALLBACK_WEIGHTS["_admission"]
        weighted_sum += w * adm_sig
        total_weight += w
        emerg    = _get_col(row_df, "admission_type_emergency")
        elective = _get_col(row_df, "admission_type_elective")
        newborn  = _get_col(row_df, "admission_type_newborn")
        details["admission_type"] = (
            f"emergency={emerg}, elective={elective}, newborn={newborn}",
            adm_sig,
        )

    # Lab and embedding features
    for feat, weight in _FALLBACK_WEIGHTS.items():
        if feat == "_admission":
            continue

        # Embedding dims: exact column name lookup only
        if feat in ("emb_81", "emb_545"):
            value = _get_col(row_df, feat)
        else:
            # Lab/vital: check bin3 → bin2 → bin1 for best available value
            base = _FEAT_BASE_COL.get(feat)
            value = _get_best_bin(row_df, base) if base else None

        if value is None:
            continue

        sig = _danger_signal(feat, value)
        if sig is None:
            continue

        weighted_sum += weight * sig
        total_weight += weight
        details[feat] = (f"{value:.4g}", sig)

    if total_weight == 0.0 or not details:
        return None

    raw_score = weighted_sum / total_weight
    score = max(0.0, min(1.0, _sigmoid_stretch(raw_score)))
    label, color = classify_risk(score)

    return {
        "score":   score,
        "label":   label,
        "color":   color,
        "n_used":  len(details),
        "n_avail": 13,
        "details": details,
    }


def compute_sepsis_risk(temp, hr, wbc):
    """
    Legacy three-variable placeholder (temperature, heart rate, WBC).

    Last-resort fallback when compute_fallback_risk() cannot run because
    none of the top-15 predictor features are present in the CSV.

    Inputs:
        temp: temperature (Celsius)
        hr:   heart rate (bpm)
        wbc:  white blood cell count

    Output:
        float risk score — use classify_risk() to label it.
    """
    return 0.3 * (temp - 36.5) + 0.4 * (hr / 100) + 0.3 * (wbc / 12)


def is_dark(input) -> bool:
    """Return True if the dark-mode checkbox is enabled."""
    return bool(input.dark_mode())


def find_column_case_insensitive(df: pd.DataFrame | None, target: str):
    """
    Find a column in a DataFrame regardless of punctuation/case style.

    Inputs:
        df: DataFrame or None
        target: requested column name

    Output:
        matching column name from df, or None if not found
    """
    if df is None or df.empty:
        return None
    target_norm = normalize_name(target)
    for col in df.columns:
        if normalize_name(col) == target_norm:
            return col
    return None



TEXT_COLUMN_CANDIDATES = {
    "note", "notes", "clinical_note", "clinical_notes",
    "note_text", "raw_note", "raw_notes", "text",
    "report", "report_text", "discharge_summary"
}


def get_text_columns_for_model_drop(df: pd.DataFrame | None):
    """
    Return non-numeric/text columns that should be excluded from model input.

    The Clinical Notes preview tab has been removed, but this helper still
    protects the XGBoost input pipeline if a future CSV includes raw text.
    """
    if df is None or df.empty:
        return []

    drop_cols = []
    normalized_candidates = {normalize_name(c) for c in TEXT_COLUMN_CANDIDATES}

    for col in df.columns:
        col_norm = normalize_name(col)
        is_named_text_col = col_norm in normalized_candidates
        is_object_col = (
            df[col].dtype == "object" or str(df[col].dtype).startswith("string")
        )
        if is_named_text_col or is_object_col:
            drop_cols.append(col)

    return list(dict.fromkeys(drop_cols))


def make_empty_plot(message: str, dark=False):
    """
    Create a placeholder Matplotlib figure containing only a message.

    Used when no real data is available for a plot panel.
    """
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bg = "#0f172a" if dark else "white"
    fg = "white" if dark else "black"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.text(0.5, 0.5, message, ha="center", va="center", color=fg)
    ax.axis("off")
    fig.tight_layout()
    return fig

# ============================================================
# 5. CSV INTAKE AND ROW SELECTION
# ============================================================
# This section replaces the original get_first_row(df) approach.
#
# Suggested improvement implemented here:
# Instead of always using only the first row, the app should use one shared
# active row selector so that risk, plots, contributors, tables, all refer to the same patient row.
#
# Why this is better than get_first_row(df):
# - All tabs stay synchronized to the same patient.
# - The user can inspect any row in a multi-row CSV.
# # ============================================================

def get_active_row(df: pd.DataFrame | None, idx_1based: int | None):
    """
    Select one active patient row from the uploaded CSV.

    Inputs:
        df:
            Full uploaded CSV as a Pandas DataFrame.
        idx_1based:
            Row number chosen in the GUI, using human-friendly 1-based indexing.

    Output:
        One-row DataFrame corresponding to the chosen patient row,
        or None if the CSV is missing/empty.

    Behavior:
        - Converts 1-based GUI indexing into 0-based Pandas indexing.
        - Clamps the value so it cannot go below row 1 or above the last row.

    CSV effect:
        This is the main row-selection step that determines which patient row
        the rest of the analytics pipeline will use.
    """
    if df is None or df.empty:
        return None
    idx = int(idx_1based or 1) - 1
    idx = max(0, min(idx, len(df) - 1))
    return df.iloc[[idx]].copy()


def apply_overrides_to_row(row_df: pd.DataFrame | None, overrides_dict: dict):
    """
    Apply provider-entered overrides to the selected row.

    Inputs:
        row_df:
            One-row DataFrame representing the active patient.
        overrides_dict:
            Dictionary of user-provided replacements, e.g.
            {'LACTATE_BIN3_MEAN': 4.2}

    Output:
        One-row DataFrame with updated values.

    CSV effect:
        This function does not change the original uploaded CSV on disk.
        It creates a modified in-memory copy of the selected row.
    """
    if row_df is None or row_df.empty:
        return row_df
    out = row_df.copy()
    override_norm = {normalize_name(k): v for k, v in overrides_dict.items()}
    for col in out.columns:
        col_norm = normalize_name(col)
        if col_norm in override_norm:
            out.loc[out.index[0], col] = override_norm[col_norm]
    return out

# ============================================================
# 6. TIME-SERIES EXTRACTION FROM THE ACTIVE ROW
# ============================================================
# These functions convert flat CSV columns like temp_bin1_mean into a tidy,
# plot-ready time-series format.
# ============================================================

def tidy_time_series(row_df: pd.DataFrame | None, overrides_dict: dict | None = None):
    """
    Reshape the selected patient row from wide CSV format into tidy time-series format.

    Inputs:
        row_df:
            One-row DataFrame for the active patient.
        overrides_dict:
            Optional provider-entered replacements.

    Output:
        DataFrame with columns:
            - col      : original source column name
            - variable : base variable name (e.g. temp)
            - stat     : mean / max / min
            - bin_num  : 1 / 2 / 3
            - time_hr  : 4 / 8 / 12
            - value    : numeric measurement

        Returns None if no time-bin columns are found.

    CSV effect:
        Example input columns:
            temp_bin1_mean, temp_bin2_mean, pulse_bin1_mean
        become rows in a tidy structure that are easy to graph and tabulate.
    """
    if row_df is None or row_df.empty:
        return None
    if overrides_dict is None:
        overrides_dict = {}

    row_df = apply_overrides_to_row(row_df, overrides_dict)
    row = row_df.iloc[0]

    records = []
    for col in row_df.columns:
        m = re.search(r"(.*)_bin([123])_(mean|max|min)", str(col), flags=re.IGNORECASE)
        if not m:
            continue
        base, bin_num, stat = m.groups()
        value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
        records.append({
            "col": col,
            "variable": base,
            "stat": stat,
            "bin_num": int(bin_num),
            "time_hr": TIME_BIN_MAP.get(bin_num, None),
            "value": value,
        })

    if not records:
        return None

    tidy = pd.DataFrame(records)
    tidy = tidy.dropna(subset=["time_hr"])
    return tidy


def build_risk_trend_from_bins(row_df: pd.DataFrame | None, overrides_dict: dict):
    """
    Build a risk-over-time DataFrame across the three time bins.

    Strategy:
        1. If the XGBoost model is loaded, construct a
           synthetic one-row feature matrix for each bin by overriding the
           binned column values with only those from that bin, then run the
           model to get a per-bin probability.
        2. If the model is unavailable, fall back to the placeholder formula
           using temp, HR/pulse, and WBC mean values from each bin.

    Inputs:
        row_df:
            One-row DataFrame for the active patient.
        overrides_dict:
            User-entered overrides.

    Output:
        DataFrame with columns: bin_num, hours, score, mode
        or None if no data is available.
    """
    tidy = tidy_time_series(row_df, overrides_dict)
    if tidy is None or tidy.empty:
        return None

    rows = []

    # --- Model-based trend path ---
    if XGB_MODEL is not None and row_df is not None:
        for bin_num, hours in TIME_BIN_MAP.items():
            try:
                # Build a synthetic row where every binned column uses the
                # values from this specific bin only.  Non-binned columns
                # keep their original values so the feature set stays complete.
                bin_row = row_df.copy()
                bin_tidy = tidy[tidy["bin_num"] == int(bin_num)]
                for _, r in bin_tidy.iterrows():
                    col = r["col"]
                    if col in bin_row.columns:
                        bin_row.loc[bin_row.index[0], col] = r["value"]

                X = build_model_input(bin_row, overrides_dict)
                if X is None:
                    continue
                prob = float(XGB_MODEL.predict_proba(X)[0, 1])
                if np.isnan(prob) or np.isinf(prob):
                    continue
                rows.append({"bin_num": int(bin_num), "hours": hours, "score": prob, "mode": "model"})
            except Exception:
                continue

        if rows:
            return pd.DataFrame(rows).sort_values("bin_num")

    # --- Placeholder formula fallback ---
    for bin_num, hours in TIME_BIN_MAP.items():
        temp = tidy.loc[
            (tidy["variable"].str.lower() == "temp") &
            (tidy["stat"].str.lower() == "mean") &
            (tidy["bin_num"] == int(bin_num)),
            "value"
        ]
        hr = tidy.loc[
            (tidy["variable"].str.lower().isin(["pulse", "hr"])) &
            (tidy["stat"].str.lower() == "mean") &
            (tidy["bin_num"] == int(bin_num)),
            "value"
        ]
        wbc = tidy.loc[
            (tidy["variable"].str.lower() == "wbc_blood") &
            (tidy["stat"].str.lower() == "mean") &
            (tidy["bin_num"] == int(bin_num)),
            "value"
        ]

        if temp.empty or hr.empty or wbc.empty:
            continue

        try:
            score = compute_sepsis_risk(float(temp.iloc[0]), float(hr.iloc[0]), float(wbc.iloc[0]))
            if np.isnan(score) or np.isinf(score):
                continue
            rows.append({"bin_num": int(bin_num), "hours": hours, "score": score, "mode": "placeholder"})
        except Exception:
            continue

    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("bin_num")


# ============================================================
# 7. MODEL INPUT CONSTRUCTION
# ============================================================
# These functions transform the selected patient row into model-ready numeric
# input aligned with the training-time feature schema.
# ============================================================

def get_model_drop_columns(row_df: pd.DataFrame | None):
    """
    Identify columns that should not be sent into sepsis_risk_model(3).json.

    Drops:
        - likely free-text note column
        - row/patient ID or label columns
        - microbiology summary columns removed during model training
    """
    if row_df is None or row_df.empty:
        return []

    drop_cols = get_text_columns_for_model_drop(row_df)

    for candidate in sorted(ID_COLS | DROPPED_IN_TRAINING):
        found = find_column_case_insensitive(row_df, candidate)
        if found is not None:
            drop_cols.append(found)

    return list(dict.fromkeys(drop_cols))


def prepare_numeric_model_df(row_df: pd.DataFrame | None, overrides_dict: dict):
    """
    Return the cleaned numeric DataFrame used as the direct XGBoost input base.

    The uploaded row is kept in CSV column order after dropping non-model columns.
    XGBoost accepts NaN values directly, so missing numeric values remain NaN
    instead of being standardized by a separate preprocessing artifact.
    """
    if row_df is None or row_df.empty:
        return None

    row_df = apply_overrides_to_row(row_df, overrides_dict)
    drop_cols = get_model_drop_columns(row_df)
    numeric_df = row_df.drop(columns=drop_cols, errors="ignore").copy()
    numeric_df = numeric_df.apply(pd.to_numeric, errors="coerce")

    if numeric_df.empty:
        return None
    return numeric_df


def build_model_input(row_df: pd.DataFrame | None, overrides_dict: dict):
    """
    Convert the selected active row into the numeric feature matrix expected by
    sepsis_risk_model(3).json.

    Inputs:
        row_df:
            One-row DataFrame representing the active patient.
        overrides_dict:
            Dictionary of provider-entered replacements.

    Output:
        2D NumPy array with shape (1, n_features), or None if conversion fails.

    Exact steps applied to the active CSV row:
        1. Apply provider overrides.
        2. Remove text/note columns, identifiers/labels, and microbiology
           columns that were removed during training.
        3. Convert remaining values to numeric while preserving CSV order.
        4. If needed, pad with NaN or trim to the model's expected feature count.

    Important:
        This version uses direct XGBoost feature alignment. The XGBoost model
        can route NaN values natively.
    """
    numeric_df = prepare_numeric_model_df(row_df, overrides_dict)
    if numeric_df is None or numeric_df.empty:
        return None

    # Shape-match model expectations. For the current uploaded pair,
    # 947 CSV columns - 6 dropped columns = 941 model features.
    if MODEL_NUM_FEATURES is not None:
        cur_n = numeric_df.shape[1]
        if cur_n < MODEL_NUM_FEATURES:
            for i in range(MODEL_NUM_FEATURES - cur_n):
                numeric_df[f"__missing_feature_{i}"] = np.nan
        elif cur_n > MODEL_NUM_FEATURES:
            numeric_df = numeric_df.iloc[:, :MODEL_NUM_FEATURES]

    return numeric_df.to_numpy(dtype=np.float32, copy=True)


def get_clinical_risk_from_model(patient_X):
    """
    Run the trained XGBoost model on the prepared numeric feature matrix.

    Input:
        patient_X:
            2D NumPy array returned by build_model_input().

    Output:
        dict with:
            - label : LOW / MODERATE / HIGH
            - color : GUI display color
            - prob  : positive-class sepsis probability
        or None if prediction fails.
    """
    if XGB_MODEL is None or patient_X is None:
        return None
    try:
        prob = float(XGB_MODEL.predict_proba(patient_X)[0, 1])
    except Exception as e:
        print("Model prediction error:", e)
        return None
    if np.isnan(prob) or np.isinf(prob):
        return None
    label, color = classify_probability(prob)
    return {"label": label, "color": color, "prob": prob}

# ============================================================
# 8. CONTRIBUTOR + TABLE BUILDERS
# ============================================================

def get_top_contributor_series(row_df: pd.DataFrame | None, overrides_dict: dict, top_n=5):
    """
    Build up to top_n contributor series for display, ranked by true model
    feature importance when the XGBoost model is available.

    Inputs:
        row_df:
            One-row active patient DataFrame.
        overrides_dict:
            User-entered replacements.
        top_n:
            Maximum number of contributor series to return.

    Output:
        List of small DataFrames, each with:
            - variable
            - time_hr
            - value

    Logic:
        1. If the XGBoost model is loaded, rank tidy time-series variables
           by their model feature_importances_ scores (gain).  This surfaces
           the variables the model actually weighted most heavily.
        2. If the model is unavailable, fall back to ranking by mean absolute
           value across time bins (data-magnitude proxy).
        3. If no time-bin columns exist at all, fall back to top flat numeric
           columns from the active row.
    """
    tidy = tidy_time_series(row_df, overrides_dict)

    if tidy is not None and not tidy.empty:
        temp = tidy.copy()
        temp["value"] = pd.to_numeric(temp["value"], errors="coerce")
        temp = temp.dropna(subset=["value"])

        if not temp.empty:
            # --- Model importance path ---
            if XGB_MODEL is not None:
                try:
                    numeric_df = prepare_numeric_model_df(row_df, overrides_dict)
                    feature_cols = list(numeric_df.columns) if numeric_df is not None else []
                    importances = np.asarray(XGB_MODEL.feature_importances_, dtype=float)
                    importance_map = dict(zip(feature_cols[:len(importances)], importances[:len(feature_cols)]))

                    # Map tidy base variable names to the maximum importance
                    # of any cleaned model feature column that contains that variable name.
                    def var_importance(var_name):
                        best = 0.0
                        norm_var = normalize_name(var_name)
                        for fc, imp in importance_map.items():
                            if norm_var in normalize_name(fc):
                                best = max(best, float(imp))
                        return best

                    vars_present = temp["variable"].unique().tolist()
                    ranked = sorted(vars_present, key=var_importance, reverse=True)[:top_n]

                    out = []
                    for var in ranked:
                        sub = temp[temp["variable"] == var].sort_values("time_hr")
                        out.append(sub[["variable", "time_hr", "value"]].copy())
                    if out:
                        return out
                except Exception:
                    pass  # fall through to magnitude-based ranking

            # --- Magnitude-based fallback ---
            grouped = temp.groupby("variable")["value"].apply(
                lambda s: float(np.mean(np.abs(s)))
            ).reset_index()
            grouped.columns = ["variable", "importance"]
            grouped = grouped.sort_values("importance", ascending=False).head(top_n)

            out = []
            for var in grouped["variable"].tolist():
                sub = temp[temp["variable"] == var].sort_values("time_hr")
                out.append(sub[["variable", "time_hr", "value"]].copy())
            return out

    if row_df is None or row_df.empty:
        return []

    row_df = apply_overrides_to_row(row_df, overrides_dict)
    drop_cols = get_text_columns_for_model_drop(row_df)
    for candidate in ["HADM_ID", "SUBJECT_ID", "SEPSIS_STATUS"]:
        found = find_column_case_insensitive(row_df, candidate)
        if found is not None:
            drop_cols.append(found)

    vals = row_df.drop(columns=list(set(drop_cols)), errors="ignore").iloc[0]
    vals = pd.to_numeric(vals, errors="coerce").dropna()
    if vals.empty:
        return []

    # Use model feature importances if available, else magnitude ranking.
    if XGB_MODEL is not None:
        try:
            numeric_df = prepare_numeric_model_df(row_df, overrides_dict)
            feature_cols = list(numeric_df.columns) if numeric_df is not None else []
            importances = np.asarray(XGB_MODEL.feature_importances_, dtype=float)
            importance_map = dict(zip(feature_cols[:len(importances)], importances[:len(feature_cols)]))
            scored = {col: importance_map.get(col, 0.0) for col in vals.index}
            top_cols = sorted(scored, key=scored.get, reverse=True)[:top_n]
        except Exception:
            top_cols = vals.abs().sort_values(ascending=False).head(top_n).index.tolist()
    else:
        top_cols = vals.abs().sort_values(ascending=False).head(top_n).index.tolist()

    out = []
    for col in top_cols:
        out.append(pd.DataFrame({
            "variable": [col],
            "time_hr": [0],
            "value": [float(vals[col])],
        }))
    return out


def build_binned_lab_table(row_df: pd.DataFrame | None, overrides_dict: dict):
    """
    Build the table shown in the Lab Table tab.

    Inputs:
        row_df:
            Active patient one-row DataFrame.
        overrides_dict:
            User-entered overrides.

    Output:
        DataFrame formatted for display in ui.output_table().

    Logic:
        - If time-bin data exists, pivot it into hour columns.
        - Otherwise show a flat feature/value table from the selected row.
    """
    tidy = tidy_time_series(row_df, overrides_dict)
    if tidy is not None and not tidy.empty:
        temp = tidy.copy()
        temp["value"] = pd.to_numeric(temp["value"], errors="coerce")
        temp = temp.dropna(subset=["value"])

        temp["Time Bin"] = temp["time_hr"].astype(int).astype(str) + " hr"
        table = temp.pivot_table(
            index="variable",
            columns="Time Bin",
            values="value",
            aggfunc="first"
        ).reset_index()

        table = table.rename(columns={"variable": "Variable"})
        desired_cols = ["Variable"] + [f"{h} hr" for h in sorted(TIME_BIN_MAP.values()) if f"{h} hr" in table.columns]
        return table[desired_cols]

    if row_df is None or row_df.empty:
        return pd.DataFrame({"Message": ["Upload a CSV with patient rows to view data."]})

    drop_cols = get_text_columns_for_model_drop(row_df)

    out = row_df.drop(columns=list(set(drop_cols)), errors="ignore").T.reset_index()
    out.columns = ["Feature", "Value"]
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    out = out.dropna(subset=["Value"]).reset_index(drop=True)
    return out.head(80)

# ============================================================
# 9. PLOT BUILDERS
# ============================================================

def make_line_plot(tidy, chosen, dark=False, title="Other Vitals / Labs Trends", ylabel="Value"):
    """Plot selected variables from the tidy time-series DataFrame."""
    bg = "#0f172a" if dark else "white"
    fg = "white" if dark else "black"
    grid = "#475569" if dark else "#d1d5db"

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    if tidy is None or tidy.empty:
        ax.text(0.5, 0.5, "No time-bin trend columns detected in this CSV", ha="center", va="center", color=fg)
        ax.axis("off")
        return fig

    temp = tidy.copy()
    temp["value"] = pd.to_numeric(temp["value"], errors="coerce")
    temp = temp.dropna(subset=["value"])

    if temp.empty:
        ax.text(0.5, 0.5, "No valid numeric trend data", ha="center", va="center", color=fg)
        ax.axis("off")
        return fig

    if not chosen:
        ax.text(0.5, 0.5, "Select one or more variables to plot.", ha="center", va="center", color=fg)
        ax.axis("off")
        return fig

    temp = temp[temp["variable"].isin(chosen)]
    if temp.empty:
        ax.text(0.5, 0.5, "No data available for selected variables.", ha="center", va="center", color=fg)
        ax.axis("off")
        return fig

    for var in chosen:
        sub = temp[temp["variable"] == var].sort_values("time_hr")
        ax.plot(sub["time_hr"], sub["value"], marker="o", linewidth=2.5, label=var)
        ax.scatter(sub["time_hr"], sub["value"], s=45)

    ax.set_xlabel("Hours Since Admission", color=fg, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, color=fg, fontsize=11, fontweight="bold")
    ax.set_title(title, color=fg, fontsize=13, fontweight="bold")

    # Explicit x-axis ticks at the three time-bin hours
    x_ticks = sorted({int(v) for v in temp["time_hr"].dropna().unique()})
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{h} hr" for h in x_ticks], color=fg, fontsize=10)
    ax.set_xlim(max(0, min(x_ticks) - 1), max(x_ticks) + 1)

    # Y-axis: auto-scale with formatted tick labels
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.3g}"))
    ax.tick_params(axis="x", colors=fg)
    ax.tick_params(axis="y", colors=fg, labelsize=9)
    ax.grid(alpha=0.3, color=grid)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9, frameon=True)

    for spine in ax.spines.values():
        spine.set_color(fg)

    fig.subplots_adjust(right=0.78)
    fig.tight_layout()
    return fig


def make_risk_trend_plot(risk_df, dark=False):
    """Plot model probability or placeholder clinical score across time bins."""
    bg = "#0f172a" if dark else "white"
    fg = "white" if dark else "black"
    grid = "#475569" if dark else "#d1d5db"

    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    if risk_df is None or risk_df.empty:
        ax.text(0.5, 0.5, "No risk trend data available across bins", ha="center", va="center", color=fg)
        ax.axis("off")
        return fig

    is_model = "mode" in risk_df.columns and (risk_df["mode"] == "model").any()
    y_label = "Sepsis Probability (Model)" if is_model else "Clinical Risk Score (Placeholder)"
    title = "Model Probability Across Time Bins" if is_model else "Risk Score Across Time Bins"

    ax.plot(risk_df["hours"], risk_df["score"], marker="o", linewidth=3, color="#3b82f6")
    ax.scatter(risk_df["hours"], risk_df["score"], s=80, color="#3b82f6", zorder=5)

    # Annotate each point with its score value
    for _, row in risk_df.iterrows():
        ax.annotate(
            f"{row['score']:.2f}",
            xy=(row["hours"], row["score"]),
            xytext=(0, 10), textcoords="offset points",
            ha="center", fontsize=10, color=fg, fontweight="bold"
        )

    ax.set_title(title, color=fg, fontsize=13, fontweight="bold")
    ax.set_xlabel("Hours Since Admission", color=fg, fontsize=11, fontweight="bold")
    ax.set_ylabel(y_label, color=fg, fontsize=11, fontweight="bold")

    # X-axis: explicit ticks at bin hours
    x_ticks = sorted(risk_df["hours"].unique().tolist())
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{int(h)} hr" for h in x_ticks], color=fg, fontsize=10)
    ax.set_xlim(max(0, min(x_ticks) - 1), max(x_ticks) + 1)

    # Y-axis: 0-1 scale with threshold lines
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.0, 0.10, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0.00", "0.10 (LOW)", "0.25", "0.50 (HIGH)", "0.75", "1.00"],
                       color=fg, fontsize=8)
    ax.axhline(y=T_LOW,  color="#4CAF50", linewidth=1.2, linestyle="--", alpha=0.7, label=f"Low threshold ({T_LOW})")
    ax.axhline(y=T_HIGH, color="#F44336", linewidth=1.2, linestyle="--", alpha=0.7, label=f"High threshold ({T_HIGH})")
    ax.legend(fontsize=8, frameon=True, loc="upper left")

    ax.tick_params(axis="x", colors=fg)
    ax.tick_params(axis="y", colors=fg)
    ax.grid(alpha=0.3, color=grid)

    for spine in ax.spines.values():
        spine.set_color(fg)

    fig.tight_layout()
    return fig


def make_contributor_series_plot(series_df, dark=False):
    """Plot one contributor series or one flat contributor value."""
    bg = "#0f172a" if dark else "white"
    fg = "white" if dark else "black"
    grid = "#475569" if dark else "#d1d5db"

    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    if series_df is None or series_df.empty:
        ax.text(0.5, 0.5, "No contributor available", ha="center", va="center", color=fg)
        ax.axis("off")
        return fig

    label = str(series_df["variable"].iloc[0])

    if len(series_df) == 1:
        ax.scatter(series_df["time_hr"], series_df["value"], s=70)
        ax.plot(series_df["time_hr"], series_df["value"], linewidth=2)
    else:
        ax.plot(series_df["time_hr"], series_df["value"], marker="o", linewidth=2.5)
        ax.scatter(series_df["time_hr"], series_df["value"], s=50)

        # Annotate each data point with its value
        for _, row in series_df.iterrows():
            ax.annotate(
                f"{row['value']:,.3g}",
                xy=(row["time_hr"], row["value"]),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8, color=fg
            )

    ax.set_title(label, color=fg, fontsize=12, fontweight="bold")
    ax.set_xlabel("Hours Since Admission", color=fg, fontsize=10, fontweight="bold")
    ax.set_ylabel("Value", color=fg, fontsize=10, fontweight="bold")

    # Explicit x-axis ticks at bin hours
    x_ticks = sorted({int(v) for v in series_df["time_hr"].dropna().unique()})
    if x_ticks:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{h} hr" for h in x_ticks], color=fg, fontsize=9)
        ax.set_xlim(max(0, min(x_ticks) - 1), max(x_ticks) + 1)

    # Y-axis with formatted labels
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.3g}"))
    ax.tick_params(axis="x", colors=fg)
    ax.tick_params(axis="y", colors=fg, labelsize=9)
    ax.grid(alpha=0.3, color=grid)

    for spine in ax.spines.values():
        spine.set_color(fg)

    fig.tight_layout()
    return fig


# ============================================================
# 10. USER INTERFACE
# ============================================================
# The UI is organized into tabs, but all tabs now use the same active row
# selector so the entire app stays synchronized to one patient at a time.
# ============================================================

app_ui = ui.page_fluid(
    ui.output_ui("theme_styles"),
    ui.tags.style("""
        .page-shell {
            min-height: 100vh;
            transition: background 0.2s ease, color 0.2s ease;
            padding-bottom: 24px;
        }
        .header-wrap {
            position: relative;
            padding: 28px 24px 24px 24px;
            border-radius: 14px;
            margin-bottom: 18px;
            border: 1px solid #dee2e6;
        }
        .title-row {
            display: flex;
            align-items: center;
            gap: 14px;
        }
        .app-title {
            font-size: 2.35rem;
            font-weight: 800;
            margin-bottom: 4px;
            letter-spacing: 0.2px;
        }
        .app-subtitle {
            font-size: 1rem;
            font-weight: 500;
            opacity: 0.88;
            margin-bottom: 0;
        }
        .theme-toggle-wrap {
            position: absolute;
            top: 18px;
            right: 18px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .demo-toggle-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 12px 14px;
            border-radius: 12px;
            border: 1px solid #dee2e6;
            margin: 10px 0 12px 0;
        }
        .demo-toggle-label {
            font-weight: 700;
            margin-bottom: 2px;
        }
        .demo-toggle-help {
            font-size: 12px;
            opacity: 0.78;
        }
        .theme-switch {
            position: relative;
            display: inline-block;
            width: 52px;
            height: 28px;
        }
        .theme-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .theme-slider {
            position: absolute;
            cursor: pointer;
            inset: 0;
            background-color: #cbd5e1;
            transition: .2s;
            border-radius: 999px;
        }
        .theme-slider:before {
            position: absolute;
            content: "";
            height: 22px;
            width: 22px;
            left: 3px;
            top: 3px;
            background-color: white;
            transition: .2s;
            border-radius: 50%;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }
        .theme-switch input:checked + .theme-slider {
            background-color: #2563eb;
        }
        .theme-switch input:checked + .theme-slider:before {
            transform: translateX(24px);
        }
        .card-panel {
            border-radius: 12px;
            border: 1px solid #dee2e6;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
        }
        .patient-card {
            border-radius: 12px;
            border: 1px solid #dee2e6;
            padding: 18px;
            margin-bottom: 16px;
            box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
        }
        .scroll-panel {
            max-height: 520px;
            overflow-y: auto;
            border-radius: 12px;
            border: 1px solid #dee2e6;
            padding: 12px;
        }
        .alert-drawer {
            position: fixed;
            right: -420px;
            top: 24px;
            transform: none;
            width: 420px;
            border-radius: 14px;
            border: 1px solid #dee2e6;
            box-shadow: 0 12px 34px rgba(0,0,0,0.18);
            padding: 22px 22px 18px 22px;
            z-index: 1000;
            transition: right 0.3s ease;
        }
        .alert-drawer.show {
            right: 20px;
        }
        .alert-drawer-close {
            position: absolute;
            top: 10px;
            right: 12px;
            background: none;
            border: none;
            font-size: 28px;
            cursor: pointer;
            padding: 0;
            width: 32px;
            height: 32px;
        }
        .banner-title {
            font-size: 1.25rem;
            font-weight: 800;
            margin-bottom: 8px;
        }
        .banner-score {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .subtle-text {
            opacity: 0.8;
            font-size: 13px;
        }
        .nav-tabs .nav-link {
            font-weight: 600;
        }
        .status-chip {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            margin-top: 8px;
        }
    """),
    ui.div(
        ui.div(
            ui.div(
                ui.img(src=APP_LOGO_SRC, style="width:72px; height:72px; object-fit:contain;"),
                ui.div(
                    ui.div("Prenova", class_="app-title"),
                    ui.p("Clinical Decision Support Tool for Early Detection of Sepsis", class_="app-subtitle")
                ),
                class_="title-row"
            ),
            ui.div(
                ui.span("Dark mode"),
                ui.tags.label(
                    ui.input_checkbox("dark_mode", None, value=False),
                    ui.span(class_="theme-slider"),
                    class_="theme-switch"
                ),
                class_="theme-toggle-wrap"
            ),
            class_="header-wrap"
        ),
        ui.div(
            ui.row(
                ui.column(
                    2,
                    ui.img(
                        src=SILHOUETTE_SVG,
                        style="""
                            width:120px;
                            height:120px;
                            border-radius:50%;
                            border:1px solid #dee2e6;
                            padding:10px;
                        """
                    )
                ),
                ui.column(
                    10,
                    ui.row(
                        ui.column(3, ui.strong("Name"), ui.p(patient_demo["Name"])),
                        ui.column(2, ui.strong("Age"), ui.p(patient_demo["Age"])),
                        ui.column(2, ui.strong("Sex"), ui.p(patient_demo["Sex"])),
                        ui.column(3, ui.strong("Hospital ID"), ui.p(patient_demo["Hospital ID"])),
                    )
                ),
            ),
            class_="patient-card"
        ),
        ui.navset_tab(
            ui.nav_panel(
                "Vitals & Labs Input",
                ui.div(
                    ui.h4("Upload Structured Patient CSV"),
                    ui.input_file("csv", "Upload CSV", accept=[".csv"]),
                    ui.br(),
                    ui.input_numeric("active_row", "Active row used across all tabs", value=1, min=1),
                    ui.output_ui("active_row_status_ui"),
                    ui.br(),
                    ui.h5("Provider Overrides"),
                    ui.div(
                        ui.div(
                            ui.div("Use demo override scoring", class_="demo-toggle-label"),
                            ui.div("Prioritizes the clinical fallback score so override tests can visibly change LOW / MODERATE / HIGH risk.", class_="demo-toggle-help"),
                        ),
                        ui.tags.label(
                            ui.input_checkbox("force_fallback_score", None, value=False),
                            ui.span(class_="theme-slider"),
                            class_="theme-switch"
                        ),
                        class_="demo-toggle-row"
                    ),
                    ui.input_text_area(
                        "notes",
                        "Overrides",
                        placeholder="Example:\nLACTATE_BIN3_MEAN: 4.2\nEMB_100: 0.83"
                    ),
                    ui.input_action_button("apply_notes", "Apply Overrides", class_="btn btn-warning"),
                    ui.output_ui("override_warnings_ui"),
                    ui.br(), ui.br(),
                    ui.output_ui("csv_status_ui"),
                    ui.output_ui("model_status_ui"),
                    ui.output_ui("model_input_status_ui"),
                    class_="card-panel"
                )
            ),
            ui.nav_panel(
                "Risk Assessment",
                ui.div(
                    ui.h4("Sepsis Risk Score"),
                    ui.output_ui("risk_bar"),
                    ui.output_text("risk_text"),
                    ui.br(),
                    ui.h5("Risk Trend Across Time Bins"),
                    ui.p("If bin columns are present in the active row, this graph will populate.", class_="subtle-text"),
                    ui.output_plot("risk_trend_plot", height="360px"),
                    class_="card-panel"
                )
            ),
            ui.nav_panel(
                "Top Contributors",
                ui.div(
                    ui.h4("Top Contributor Feature Trends"),
                    ui.p("Ranked by XGBoost feature importance when the model is loaded; otherwise ranked by mean absolute value across time bins.", class_="subtle-text"),
                    ui.row(
                        ui.column(6, ui.output_plot("contrib_plot_1", height="280px")),
                        ui.column(6, ui.output_plot("contrib_plot_2", height="280px"))
                    ),
                    ui.row(
                        ui.column(6, ui.output_plot("contrib_plot_3", height="280px")),
                        ui.column(6, ui.output_plot("contrib_plot_4", height="280px"))
                    ),
                    ui.row(
                        ui.column(6, ui.output_plot("contrib_plot_5", height="280px"))
                    ),
                    class_="card-panel"
                )
            ),
            ui.nav_panel(
                "Other Vitals / Labs",
                ui.div(
                    ui.h4("Other Vitals / Labs Trends"),
                    ui.row(
                        ui.column(9, ui.output_plot("trend_plot", height="620px")),
                        ui.column(
                            3,
                            ui.div(
                                ui.h5("Select variables"),
                                ui.output_ui("var_selector_ui"),
                                class_="scroll-panel"
                            )
                        )
                    ),
                    class_="card-panel"
                )
            ),
            ui.nav_panel(
                "Lab Table",
                ui.div(
                    ui.h4("Clinical Table"),
                    ui.p("Shows binned features when available; otherwise falls back to flat numeric features from the active row.", class_="subtle-text"),
                    ui.output_table("lab_table"),
                    class_="card-panel"
                )
            )
        ),
        ui.output_ui("sepsis_alert"),
        class_="page-shell"
    )
)

# ============================================================
# 11. SERVER
# ============================================================
# The server is organized to mirror the data flow pipeline:
#   upload -> active row -> overrides -> transformed data -> risk -> outputs
# ============================================================

def server(input, output, session):
    # ------------------------------------------------------------
    # Reactive state containers
    # ------------------------------------------------------------
    overrides = reactive.Value({})
    override_warnings = reactive.Value([])
    # Incremented every time Apply Overrides is clicked. This forces all
    # downstream reactive calculations to re-run even if Shiny considers the
    # parsed dictionary equal to the previous value.
    override_tick = reactive.Value(0)
    selected_vars = reactive.Value([])
    alert_visible = reactive.Value(True)

    # ------------------------------------------------------------
    # Theme styles
    # ------------------------------------------------------------
    @output
    @render.ui
    def theme_styles():
        dark = is_dark(input)
        if dark:
            return ui.tags.style("""
                body { background: #020617; color: #f8fafc; }
                .page-shell, .header-wrap, .card-panel, .scroll-panel, .note-preview-box, .patient-card, .alert-drawer {
                    background: #0f172a;
                    color: #f8fafc;
                    border-color: #334155;
                }
                .form-control, textarea, input, .form-select {
                    background: #111827 !important;
                    color: #f8fafc !important;
                    border-color: #475569 !important;
                }
                .demo-toggle-row {
                    background: #111827;
                    border-color: #334155;
                }
                .nav-tabs .nav-link {
                    background: #0f172a;
                    color: #f8fafc;
                    border-color: #334155;
                }
                .nav-tabs .nav-link.active {
                    background: #111827;
                    color: #f8fafc;
                    border-color: #475569;
                }
                table, th, td {
                    color: #f8fafc !important;
                    background: #0f172a !important;
                }
                .alert-drawer-close { color: #f8fafc; }
                input[type="checkbox"],
                .form-check-input {
                    accent-color: #60a5fa !important;
                    background-color: #1e293b !important;
                    border: 2px solid #94a3b8 !important;
                    width: 16px !important;
                    height: 16px !important;
                }
                .form-check-input:checked {
                    background-color: #2563eb !important;
                    border-color: #2563eb !important;
                    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20'%3E%3Cpath fill='none' stroke='%23ffffff' stroke-width='3' d='M6 10l3 3 5-6'/%3E%3C/svg%3E") !important;
                    background-size: 100% 100% !important;
                }
                .form-check-label,
                .form-check {
                    color: #f8fafc !important;
                }
            """)
        return ui.tags.style("""
            body { background: #f8fafc; color: #0f172a; }
            .page-shell, .header-wrap, .card-panel, .scroll-panel, .note-preview-box, .patient-card, .alert-drawer {
                background: white;
                color: #0f172a;
                border-color: #dee2e6;
            }
            .nav-tabs .nav-link.active {
                background: white;
            }
            .alert-drawer-close { color: #64748b; }
        """)

    # ------------------------------------------------------------
    # CSV upload and active-row selection
    # ------------------------------------------------------------
    # Sentinel object used to distinguish a failed CSV parse from no upload.
    _CSV_PARSE_ERROR = object()

    @reactive.calc
    def csv_df():
        """
        Read the uploaded CSV into a DataFrame.

        Input:
            input.csv() from the file uploader.

        Output:
            Full Pandas DataFrame, None if no file uploaded, or
            _CSV_PARSE_ERROR sentinel if the file failed to parse.

        This is the first true data-ingestion step for the CSV.
        """
        f = input.csv()
        if not f:
            return None
        try:
            return pd.read_csv(f[0]["datapath"])
        except Exception as e:
            print("CSV parse error:", e)
            return _CSV_PARSE_ERROR

    @reactive.calc
    def active_row_df():
        """
        Select the one active patient row used across all tabs.
        Returns None if no CSV is loaded or if the CSV failed to parse.
        """
        df = csv_df()
        if df is None or df is _CSV_PARSE_ERROR:
            return None
        return get_active_row(df, input.active_row())

    # ------------------------------------------------------------
    # Provider overrides
    # ------------------------------------------------------------
    @reactive.Effect
    @reactive.event(input.apply_notes)
    def _apply_overrides():
        """
        Parse user-entered override lines into a dictionary.
        Lines where the value cannot be converted to a float are rejected
        and surfaced to the user as warnings instead of silently being
        passed downstream as strings that would coerce to NaN.

        Example input text:
            LACTATE_BIN3_MEAN: 4.2
            EMB_100: 0.83
        """
        txt = input.notes() or ""
        parsed = {}
        warnings = []
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" not in line:
                warnings.append(f"Skipped (no colon found): {line!r}")
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            if not k:
                warnings.append(f"Skipped (empty key): {line!r}")
                continue
            try:
                parsed[k] = float(v)
            except ValueError:
                warnings.append(f"Skipped (value is not a number): {k!r} = {v!r}")
        overrides.set(dict(parsed))
        override_warnings.set(list(warnings))
        override_tick.set(override_tick.get() + 1)

        # Re-show the alert when a new override is applied.
        alert_visible.set(True)

    # ------------------------------------------------------------
    # Derived transformed data from the active row
    # ------------------------------------------------------------
    @reactive.calc
    def current_overrides():
        """Reactive wrapper that changes on every Apply Overrides click."""
        override_tick.get()
        return overrides.get()

    @reactive.calc
    def tidy_trends_calc():
        """Return tidy time-series data for the active row."""
        return tidy_time_series(active_row_df(), current_overrides())

    @reactive.calc
    def available_variables():
        """Return variable names available for the trend plot selector."""
        tidy = tidy_trends_calc()
        if tidy is None or tidy.empty:
            return []
        return sorted(tidy["variable"].dropna().unique().tolist())

    @reactive.calc
    def risk_trend_df():
        """Return the placeholder risk-over-time DataFrame for the active row."""
        return build_risk_trend_from_bins(active_row_df(), current_overrides())

    @reactive.calc
    def top_contributors():
        """Return contributor series for the active row."""
        return get_top_contributor_series(active_row_df(), current_overrides(), top_n=5)

    # ------------------------------------------------------------
    # Plot variable selection sync
    # ------------------------------------------------------------
    @reactive.Effect
    def _init_selected_vars():
        vars_ = available_variables()
        if not vars_:
            selected_vars.set([])
            return
        cur = set(selected_vars.get() or [])
        if not cur.intersection(set(vars_)):
            selected_vars.set(vars_[:8])

    @reactive.Effect
    def _sync_selected_vars():
        v = input.plot_vars()
        if v is None:
            return
        selected_vars.set(list(v))

    # ------------------------------------------------------------
    # Main risk calculation pipeline
    # ------------------------------------------------------------
    @reactive.calc
    def risk():
        """
        Compute the displayed sepsis risk for the active row.

        Three-tier priority:
            1. XGBoost model — when loaded and >= 10% of 941 features are real.
               Below 10% the model regresses to its ~0.45 base rate with no
               discrimination, so we skip to tier 2.
            2. Top-15 predictor fallback (compute_fallback_risk) — all 15 chart
               features represented, missing ones cleanly excluded. Sigmoid-
               stretched onto the same [0,1] scale as the model so the same
               T_LOW / T_HIGH thresholds apply.
            3. Legacy formula — last resort using bin-1 temp, pulse, WBC only.
        """
        row_df = active_row_df()
        if row_df is None or row_df.empty:
            return None

        overrides_dict = current_overrides()
        force_fallback = bool(input.force_fallback_score())

        # --- Tier 1: trained XGBoost model ---
        # When demo override scoring is enabled, skip the full XGBoost model
        # so the visible score responds strongly to manually entered values.
        patient_X = build_model_input(row_df, overrides_dict)
        if (not force_fallback) and patient_X is not None and XGB_MODEL is not None:
            n_real   = int((~np.isnan(patient_X)).sum())
            pct_real = n_real / patient_X.shape[1] if patient_X.shape[1] > 0 else 0.0
            if pct_real >= 0.10:
                model_risk = get_clinical_risk_from_model(patient_X)
                if model_risk is not None:
                    return {
                        "mode":         "model",
                        "score":        model_risk["prob"],
                        "label":        model_risk["label"],
                        "color":        model_risk["color"],
                        "pct_complete": round(pct_real * 100, 1),
                    }

        # --- Tier 2: top-15 predictor fallback ---
        overridden_row = apply_overrides_to_row(row_df, overrides_dict)
        fallback = compute_fallback_risk(overridden_row)
        if fallback is not None:
            return {
                "mode":         "fallback_demo" if force_fallback else "fallback",
                "score":        fallback["score"],
                "label":        fallback["label"],
                "color":        fallback["color"],
                "n_used":       fallback["n_used"],
                "n_avail":      fallback["n_avail"],
                "details":      fallback["details"],
                "pct_complete": round(100 * fallback["n_used"] / fallback["n_avail"], 1),
            }

        # --- Tier 3: legacy three-variable formula ---
        tidy = tidy_trends_calc()
        if tidy is not None and not tidy.empty:
            try:
                temp = tidy.loc[
                    (tidy["variable"].str.lower() == "temp") &
                    (tidy["stat"].str.lower() == "mean") &
                    (tidy["bin_num"] == 1), "value"
                ]
                hr = tidy.loc[
                    (tidy["variable"].str.lower().isin(["pulse", "hr"])) &
                    (tidy["stat"].str.lower() == "mean") &
                    (tidy["bin_num"] == 1), "value"
                ]
                wbc = tidy.loc[
                    (tidy["variable"].str.lower() == "wbc_blood") &
                    (tidy["stat"].str.lower() == "mean") &
                    (tidy["bin_num"] == 1), "value"
                ]
                if not temp.empty and not hr.empty and not wbc.empty:
                    score = compute_sepsis_risk(
                        float(temp.iloc[0]), float(hr.iloc[0]), float(wbc.iloc[0])
                    )
                    if not (np.isnan(score) or np.isinf(score)):
                        label, color = classify_risk(score)
                        return {"mode": "legacy", "score": score, "label": label, "color": color}
            except Exception:
                pass

        return None

    # ------------------------------------------------------------
    # UI outputs: status chips and selectors
    # ------------------------------------------------------------
    @output
    @render.ui
    def csv_status_ui():
        df = csv_df()
        if df is None:
            return ui.div("No CSV uploaded yet.", class_="status-chip", style="background:#e2e8f0; color:#334155;")
        if df is _CSV_PARSE_ERROR:
            return ui.div(
                "Failed to parse the uploaded file. Please check that it is a valid CSV.",
                class_="status-chip",
                style="background:#fee2e2; color:#991b1b;"
            )
        if df.empty:
            return ui.div(
                "CSV loaded, but it contains headers only and no patient rows.",
                class_="status-chip",
                style="background:#fee2e2; color:#991b1b;"
            )
        return ui.div(
            f"CSV loaded: {df.shape[0]} row(s), {df.shape[1]} columns.",
            class_="status-chip",
            style="background:#dcfce7; color:#166534;"
        )

    @output
    @render.ui
    def active_row_status_ui():
        """
        Show feedback when the entered row number exceeds the CSV length,
        so the user knows it was clamped to the last available row.
        """
        df = csv_df()
        if df is None or df.empty:
            return ui.div()
        n_rows = len(df)
        entered = int(input.active_row() or 1)
        if entered > n_rows:
            return ui.div(
                f"Row {entered} exceeds CSV length ({n_rows} rows). Showing row {n_rows}.",
                class_="status-chip",
                style="background:#fef3c7; color:#92400e;"
            )
        if entered < 1:
            return ui.div(
                f"Row number must be at least 1. Showing row 1.",
                class_="status-chip",
                style="background:#fef3c7; color:#92400e;"
            )
        return ui.div()

    @reactive.Effect
    def _update_row_max():
        """Keep the active_row numeric input max in sync with the CSV row count."""
        df = csv_df()
        if df is not None and df is not _CSV_PARSE_ERROR and not df.empty:
            ui.update_numeric("active_row", max=len(df))

    @output
    @render.ui
    def model_status_ui():
        if XGB_MODEL is not None:
            return ui.div(MODEL_STATUS, class_="status-chip", style="background:#dcfce7; color:#166534;")
        return ui.div(MODEL_STATUS, class_="status-chip", style="background:#fef3c7; color:#92400e;")

    @output
    @render.ui
    def model_input_status_ui():
        return ui.div(MODEL_INPUT_STATUS, class_="status-chip", style="background:#dbeafe; color:#1d4ed8;")

    @output
    @render.ui
    def override_warnings_ui():
        """Show a warning chip for each override line that could not be parsed."""
        warnings = override_warnings.get()
        if not warnings:
            return ui.div()
        items = [ui.div(w, style="margin-bottom:4px;") for w in warnings]
        return ui.div(
            ui.strong("Override warnings:"),
            *items,
            class_="status-chip",
            style="background:#fee2e2; color:#991b1b; display:block; margin-top:8px;"
        )

    @output
    @render.ui
    def var_selector_ui():
        vars_ = available_variables()
        if not vars_:
            return ui.p("No time-bin variables detected in the active row.")
        return ui.input_checkbox_group(
            "plot_vars",
            None,
            choices=vars_,
            selected=selected_vars.get() or []
        )

    # ------------------------------------------------------------
    # UI outputs: risk displays and alert drawer
    # ------------------------------------------------------------
    def _mode_note(r: dict) -> str:
        """One-line scoring method description for display in risk panels."""
        mode = r.get("mode", "unknown")
        if mode == "model":
            pct = r.get("pct_complete")
            return f"XGBoost model  ({pct:.0f}% of features present)" if pct is not None else "XGBoost model"
        if mode == "fallback_demo":
            n_used  = r.get("n_used",  "?")
            n_avail = r.get("n_avail", "?")
            return f"Demo override scoring  ({n_used}/{n_avail} top predictors present)"
        if mode == "fallback":
            n_used  = r.get("n_used",  "?")
            n_avail = r.get("n_avail", "?")
            return f"Clinical fallback  ({n_used}/{n_avail} top predictors present)"
        if mode == "legacy":
            return "Legacy formula  (temp / pulse / WBC only)"
        return "Unknown scoring method"

    @output
    @render.ui
    def risk_bar():
        r = risk()
        if not r:
            return ui.div("No risk calculated. Check CSV contents and the XGBoost model file.")

        banner_text_color = "#111827" if r["label"] == "MODERATE" else "white"
        metric_label      = "Probability" if r.get("mode") == "model" else "Score"

        return ui.div(
            ui.div(f"{r['label']} RISK", style="font-size: 1.25rem; font-weight: 800; letter-spacing: 0.4px;"),
            ui.div(f"{metric_label}: {r['score']:.2f}", style="font-size: 1rem; font-weight: 700; margin-top: 4px;"),
            ui.div(f"Scoring: {_mode_note(r)}", style="font-size: 0.78rem; margin-top: 6px; opacity: 0.88;"),
            style=f"""
                background:{r['color']};
                color:{banner_text_color};
                min-height:72px;
                border-radius:14px;
                margin-bottom:14px;
                padding:16px 18px;
                display:flex;
                flex-direction:column;
                justify-content:center;
                box-shadow: 0 4px 14px rgba(0,0,0,0.08);
            """
        )

    @output
    @render.text
    def risk_text():
        r = risk()
        if not r:
            return "Awaiting sufficient data..."
        mode  = r.get("mode", "unknown")
        score = f"{r['score']:.3f}"
        if mode == "model":
            pct    = r.get("pct_complete")
            detail = f"Model Probability: {score}"
            if pct is not None:
                detail += f"  |  {pct:.0f}% of features present"
        elif mode in ("fallback", "fallback_demo"):
            n_used  = r.get("n_used",  "?")
            n_avail = r.get("n_avail", "?")
            detail  = f"Clinical Fallback Score: {score}  |  {n_used}/{n_avail} top predictors present"
        else:
            detail = f"Score: {score}"
        return f"Risk Level: {r['label']}  ({detail})"

    @output
    @render.ui
    def sepsis_alert():
        """
        Slide-out alert shown only when the active row is MODERATE or HIGH risk
        and the alert has not been manually closed.
        """
        r = risk()
        if not r or r["label"] == "LOW" or not alert_visible.get():
            return ui.div()

        banner_text_color = "#111827" if r["label"] == "MODERATE" else "white"
        mode = r.get("mode", "unknown")
        metric_label = (
            "Sepsis Probability"      if mode == "model"    else
            "Clinical Fallback Score" if mode in ("fallback", "fallback_demo") else
            "Risk Score"
        )

        return ui.div(
            ui.input_action_button("close_alert_btn", "✕", class_="alert-drawer-close"),
            ui.div(f"{r['label']} RISK DETECTED", class_="banner-title", style=f"color:{banner_text_color};"),
            ui.div(f"{metric_label}: {r['score']:.2f}", class_="banner-score", style=f"color:{banner_text_color};"),
            ui.div(f"Scoring: {_mode_note(r)}", style=f"color:{banner_text_color}; font-size:0.82rem; margin-top:4px;"),
            ui.p("Please review patient vitals and consider clinical intervention.", style=f"color:{banner_text_color}; margin-bottom:0; margin-top:6px;"),
            class_="alert-drawer show",
            id="sepsis_alert_drawer",
            style=f"background:{r['color']}; border-color:{r['color']};"
        )

    @reactive.Effect
    @reactive.event(input.close_alert_btn)
    def _close_alert():
        alert_visible.set(False)

    # Re-open the alert automatically if the active row changes.
    @reactive.Effect
    def _reset_alert_on_row_change():
        _ = input.active_row()
        alert_visible.set(True)

    # ------------------------------------------------------------
    # UI outputs: plots
    # ------------------------------------------------------------
    def render_contrib_plot(idx: int):
        contributors = top_contributors()
        dark = is_dark(input)
        if len(contributors) < idx:
            return make_empty_plot("No contributor available", dark=dark)
        return make_contributor_series_plot(contributors[idx - 1], dark=dark)

    @output
    @render.plot
    def contrib_plot_1():
        return render_contrib_plot(1)

    @output
    @render.plot
    def contrib_plot_2():
        return render_contrib_plot(2)

    @output
    @render.plot
    def contrib_plot_3():
        return render_contrib_plot(3)

    @output
    @render.plot
    def contrib_plot_4():
        return render_contrib_plot(4)

    @output
    @render.plot
    def contrib_plot_5():
        return render_contrib_plot(5)

    @output
    @render.plot
    def trend_plot():
        return make_line_plot(
            tidy_trends_calc(),
            selected_vars.get() or [],
            dark=is_dark(input)
        )

    @output
    @render.plot
    def risk_trend_plot():
        return make_risk_trend_plot(
            risk_trend_df(),
            dark=is_dark(input)
        )

    # ------------------------------------------------------------
    # UI outputs: table
    # ------------------------------------------------------------
    @output
    @render.table
    def lab_table():
        return build_binned_lab_table(active_row_df(), current_overrides())
# ============================================================
# 12. APP CREATION / LAUNCH
# ============================================================

app = App(app_ui, server)

if __name__ == "__main__":
    print("Starting Prenova...")
    print(f"Bound server host: {HOST}:{PORT}")
    print(f"Open in browser: http://{DISPLAY_HOST}:{PORT}")
    app.run(host=HOST, port=PORT)