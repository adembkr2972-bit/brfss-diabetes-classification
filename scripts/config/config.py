import os
from pathlib import Path


def _path_from_env(env_name: str, default: Path) -> Path:
    value = os.environ.get(env_name)
    return Path(value) if value else default


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = _path_from_env("DIABETES_DATA_DIR", PROJECT_ROOT / "data")
RAW_DIR = _path_from_env("DIABETES_RAW_DIR", DATA_DIR / "raw")
INTERIM_DIR = _path_from_env("DIABETES_INTERIM_DIR", DATA_DIR / "interim")
PROCESSED_DIR = _path_from_env("DIABETES_PROCESSED_DIR", DATA_DIR / "processed")

MODELS_DIR = _path_from_env("DIABETES_MODELS_DIR", PROJECT_ROOT / "models")
LOGS_DIR = _path_from_env("DIABETES_LOGS_DIR", PROJECT_ROOT / "logs")
RERUN_STARTED_PATH = LOGS_DIR / "approved_protocol_rerun_start.txt"

OUTPUTS_DIR = _path_from_env("DIABETES_OUTPUTS_DIR", PROJECT_ROOT / "outputs")
METRICS_DIR = _path_from_env("DIABETES_METRICS_DIR", OUTPUTS_DIR / "metrics")
PREDICTIONS_DIR = _path_from_env("DIABETES_PREDICTIONS_DIR", OUTPUTS_DIR / "predictions")
FIGURES_DIR = _path_from_env("DIABETES_FIGURES_DIR", OUTPUTS_DIR / "figures")
REPORTS_DIR = _path_from_env("DIABETES_REPORTS_DIR", OUTPUTS_DIR / "reports")

RAW_CSV_PATH = _path_from_env("DIABETES_RAW_CSV_PATH", RAW_DIR / "LLCP2024XPT" / "LLCP2024.csv")

AUDIT_REPORT_PATH = INTERIM_DIR / "audit_report.txt"
COLUMN_SUMMARY_PATH = INTERIM_DIR / "column_summary.csv"
TARGET_DISTRIBUTION_PATH = INTERIM_DIR / "target_distribution.csv"
EXCLUDED_ROWS_PATH = INTERIM_DIR / "excluded_rows_summary.csv"
BINARIZED_PARQUET_PATH = INTERIM_DIR / "dataset_binarized.parquet"
FEATURES_SELECTED_PATH = INTERIM_DIR / "dataset_features_selected.parquet"
FINAL_FEATURE_LIST_PATH = INTERIM_DIR / "final_feature_list.txt"
FEATURE_SELECTION_LOG_PATH = INTERIM_DIR / "feature_selection_log.csv"
COLUMN_TYPE_PATH = INTERIM_DIR / "column_type_assignments.csv"
NUMERIC_COLS_PATH = INTERIM_DIR / "numeric_cols.txt"
CATEGORICAL_COLS_PATH = INTERIM_DIR / "categorical_cols.txt"
PIPELINE_SCALE_PATH = INTERIM_DIR / "pipeline_scale_sensitive.joblib"
PIPELINE_TREE_PATH = INTERIM_DIR / "pipeline_tree_based.joblib"

X_TRAIN_PATH = PROCESSED_DIR / "X_train.parquet"
X_VAL_PATH = PROCESSED_DIR / "X_val.parquet"
X_TEST_PATH = PROCESSED_DIR / "X_test.parquet"
Y_TRAIN_PATH = PROCESSED_DIR / "y_train.parquet"
Y_VAL_PATH = PROCESSED_DIR / "y_val.parquet"
Y_TEST_PATH = PROCESSED_DIR / "y_test.parquet"
SPLIT_INDICES_PATH = PROCESSED_DIR / "split_indices.npz"
SPLIT_SUMMARY_PATH = PROCESSED_DIR / "split_summary.csv"

BASELINE_METRICS_PATH = METRICS_DIR / "baseline_validation_metrics.csv"
TUNED_METRICS_PATH = METRICS_DIR / "tuned_validation_metrics.csv"
FULL_VALIDATION_METRICS_PATH = METRICS_DIR / "full_validation_metrics.csv"
OVERFITTING_ANALYSIS_PATH = METRICS_DIR / "overfitting_analysis.csv"
TUNED_BASELINE_DECISION_PATH = METRICS_DIR / "tuned_vs_baseline_decision.csv"
CHOSEN_THRESHOLDS_PATH = METRICS_DIR / "chosen_thresholds.csv"
FINAL_TEST_METRICS_PATH = METRICS_DIR / "final_test_metrics.csv"
MODEL_SHORTLIST_PATH = REPORTS_DIR / "model_shortlist.csv"
MODEL_ELIMINATION_REPORT_PATH = REPORTS_DIR / "model_elimination_report.txt"
SUBGROUP_ERROR_ANALYSIS_PATH = REPORTS_DIR / "subgroup_error_analysis.csv"
FINAL_COMPARISON_TABLE_PATH = REPORTS_DIR / "final_comparison_table.csv"
FINAL_REPORT_PATH = REPORTS_DIR / "final_report.txt"

TARGET_COL = "DIABETE4"
TARGET_BINARY_COL = "target"
TARGET_KEEP_VALUES = {1: 1, 3: 0}

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

LEAKAGE_COLS = [
    "DIABAGE4",
    "PDIABTS1",
    "PREDIAB2",
    "DIABTYPE",
    "INSULIN1",
    "CHKHEMO3",
    "EYEEXAM1",
    "DIABEYE1",
    "DIABEDU1",
    "FEETSORE",
]

DROP_COLS = [
    "_STATE",
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "DISPCODE",
    "SEQNO",
    "_PSU",
    "CTELENM1",
    "PVTRESD1",
    "COLGHOUS",
    "STATERE1",
    "CELPHON1",
    "LADULT1",
    "NUMADULT",
    "RESPSLC1",
    "LANDSEX3",
    "SAFETIME",
    "CTELNUM1",
    "CELLFON5",
    "CADULT1",
    "CELLSEX3",
    "PVTRESD3",
    "CCLGHOUS",
    "CSTATE1",
    "LANDLINE",
    "HHADULT",
    "_STSTR",
    "_RAWRAKE",
    "_WT2RAKE",
    "_DUALUSE",
    "_DUALCOR",
    "_LLCPWT2",
    "_LLCPWT",
    "_STRWT",
    "_CLLCPWT",
    "_CLCWTV2",
    "_LCPWTV2",
    "MSCODE",
    "_METSTAT",
    "_URBSTAT",
    "QSTVER",
    "QSTLANG",
    "ICFQSTVR",
    "CPDEMO1C",
    "NUMPHON4",
]

FEATURES_DEMOGRAPHICS = [
    "SEXVAR",
    "_SEX",
    "_AGEG5YR",
    "_AGE65YR",
    "_AGE_G",
    "MARITAL",
    "VETERAN3",
    "CHILDREN",
    "_CHLDCNT",
]

FEATURES_SOCIOECONOMIC = [
    "EDUCA",
    "_EDUCAG",
    "INCOME3",
    "_INCOMG1",
    "EMPLOY1",
    "RENTHOM1",
]

FEATURES_BODY = [
    "_BMI5",
    "_BMI5CAT",
    "_RFBMI5",
    "HTM4",
    "WTKG3",
    "HTIN4",
]

FEATURES_GENERAL_HEALTH = [
    "GENHLTH",
    "PHYSHLTH",
    "MENTHLTH",
    "POORHLTH",
    "_RFHLTH",
    "_PHYS14D",
    "_MENT14D",
]

FEATURES_HEALTHCARE_ACCESS = [
    "PRIMINS2",
    "_HLTHPL2",
    "_HCVU654",
    "PERSDOC3",
    "MEDCOST1",
    "CHECKUP1",
]

FEATURES_PHYSICAL_ACTIVITY = [
    "EXERANY2",
    "_TOTINDA",
]

FEATURES_SMOKING = [
    "SMOKE100",
    "SMOKDAY2",
    "USENOW3",
    "_SMOKER3",
    "_RFSMOK3",
]

FEATURES_ALCOHOL = [
    "ALCDAY4",
    "AVEDRNK4",
    "DRNKANY6",
    "_DRNKWK3",
    "_RFDRHV9",
    "_RFBING6",
]

FEATURES_COMORBIDITIES = [
    "CVDINFR4",
    "CVDCRHD4",
    "CVDSTRK3",
    "ASTHMA3",
    "CHCCOPD3",
    "ADDEPEV3",
    "CHCKDNY2",
    "HAVARTH4",
    "CHCSCNC1",
    "CHCOCNC1",
    "BPHIGH6",
    "TOLDHI3",
    "CHOLMED3",
    "BPMEDS1",
    "_RFHYPE6",
    "_RFCHOL3",
    "_MICHD",
    "_DRDXAR2",
]

FEATURES_DISABILITY = [
    "DEAF",
    "BLIND",
    "DECIDE",
    "DIFFWALK",
    "DIFFDRES",
    "DIFFALON",
]

FEATURES_PREVENTIVE = [
    "FLUSHOT7",
    "PNEUVAC4",
    "_FLSHOT7",
    "_PNEUMO3",
]

FEATURES_RACE = [
    "_RACE",
    "_RACEGR3",
    "_HISPANC",
    "_MRACE1",
]

CANDIDATE_FEATURES = (
    FEATURES_DEMOGRAPHICS
    + FEATURES_SOCIOECONOMIC
    + FEATURES_BODY
    + FEATURES_GENERAL_HEALTH
    + FEATURES_HEALTHCARE_ACCESS
    + FEATURES_PHYSICAL_ACTIVITY
    + FEATURES_SMOKING
    + FEATURES_ALCOHOL
    + FEATURES_COMORBIDITIES
    + FEATURES_DISABILITY
    + FEATURES_PREVENTIVE
    + FEATURES_RACE
)

OPTIONAL_MODULE_FEATURES = [
    "ACEDEPRS",
    "ACEDRINK",
    "ACEDRUGS",
    "ACEPRISN",
    "ACEDIVRC",
    "ACEPUNCH",
    "ACEHURT1",
    "ACESWEAR",
    "ACETOUCH",
    "ACETTHEM",
    "ACEHVSEX",
    "ACEADSAF",
    "ACEADNED",
    "LSATISFY",
    "EMTSUPRT",
    "SDLONELY",
    "SDHEMPLY",
    "FOODSTMP",
    "SDHFOOD1",
    "SDHBILLS",
    "SDHUTILS",
    "SDHTRNSP",
    "HOWSAFE1",
    "CIMEMLO1",
    "CDWORRY",
    "CDDISCU1",
    "CDHOUS1",
    "CDSOCIA1",
    "CAREGIV1",
    "CRGVREL5",
    "CRGVPRB4",
    "CRGVALZD",
    "CRGVNURS",
    "CRGVPER2",
    "CRGVHOU2",
    "CRGVHRS2",
    "CRGVLNG2",
    "SSBSUGR2",
    "SSBFRUT3",
    "LASTSMK2",
    "STOPSMK2",
    "MENTCIGS",
    "MENTECIG",
    "MARIJAN1",
    "MARJSMOK",
    "MARJEAT",
    "MARJVAPE",
    "MARJDAB",
    "MARJOTHR",
    "USEMRJN4",
    "CSRVTRT3",
    "CSRVDOC1",
    "CSRVSUM",
    "CSRVRTRN",
    "CSRVINST",
    "CSRVINSR",
    "CSRVDEIN",
    "CSRVCLIN",
    "CSRVPAIN",
    "CSRVCTL2",
    "ARTHEXER",
]

CONTINUOUS_FEATURES = [
    "_BMI5",
    "PHYSHLTH",
    "MENTHLTH",
    "POORHLTH",
    "ALCDAY4",
    "AVEDRNK4",
    "_DRNKWK3",
    "HTM4",
    "WTKG3",
    "HTIN4",
    "CHILDREN",
    "_CHLDCNT",
]

ORDINAL_FEATURES = [
    "GENHLTH",
    "_AGEG5YR",
    "_AGE65YR",
    "_AGE_G",
    "_EDUCAG",
    "_INCOMG1",
    "_BMI5CAT",
    "INCOME3",
    "EDUCA",
    "CHECKUP1",
    "_PHYS14D",
    "_MENT14D",
]

# Race/ethnicity, employment, insurance, marital status, and other response
# groups have no defensible numeric order and are one-hot encoded.
MAX_MISSING_FEATURE_PCT = 40.0

MODEL_NAMES = [
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    "ExtraTrees",
    "SVM",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "MLPClassifier",
]

MODEL_DISPLAY_LABELS = {
    "SVM": "SVM (trained on a 50,000-row stratified subsample)",
}


def display_model_label(model_name: str) -> str:
    return MODEL_DISPLAY_LABELS.get(model_name, model_name)

SCALE_SENSITIVE_MODELS = ["LogisticRegression", "SVM", "MLPClassifier"]
CV_FOLDS = 5
CV_SCORING = "f1_macro"
USE_GPU = os.environ.get("DIABETES_USE_GPU", "0") == "1"
MAX_PARALLEL_JOBS = max(1, min(2, os.cpu_count() or 1))
SVM_BASELINE_MAX_ROWS = int(os.environ.get("DIABETES_SVM_BASELINE_MAX_ROWS", "50000"))
SVM_TUNING_MAX_ROWS = int(os.environ.get("DIABETES_SVM_TUNING_MAX_ROWS", "10000"))
SKIP_SVM_TUNING = os.environ.get("DIABETES_SKIP_SVM_TUNING", "0") == "1"

BRFSS_SPECIAL_VALUES_GENERAL = [7, 9, 77, 99]
BMI5_SCALE_FACTOR = 100

SPECIAL_RECODE_MAPS = {
    "ALCDAY4": {777: None, 888: 0, 999: None},
    "AVEDRNK4": {77: None, 88: 0, 99: None},
    "PHYSHLTH": {77: None, 88: 0, 99: None},
    "MENTHLTH": {77: None, 88: 0, 99: None},
    "POORHLTH": {77: None, 88: 0, 99: None},
    "CHILDREN": {88: 0, 99: None},
    "INCOME3": {77: None, 99: None},
    "PRIMINS2": {77: None, 99: None},
    "_DRNKWK3": {99900: None, 9: None},
    "GENHLTH": {7: None, 9: None},
    "_AGEG5YR": {14: None},
    "_AGE65YR": {3: None},
}

GENERAL_RECODE_7_9_COLUMNS = [
    "SEXVAR",
    "VETERAN3",
    "PERSDOC3",
    "MEDCOST1",
    "CHECKUP1",
    "EXERANY2",
    "SMOKE100",
    "SMOKDAY2",
    "USENOW3",
    "CVDINFR4",
    "CVDCRHD4",
    "CVDSTRK3",
    "ASTHMA3",
    "CHCCOPD3",
    "ADDEPEV3",
    "CHCKDNY2",
    "HAVARTH4",
    "CHCSCNC1",
    "CHCOCNC1",
    "BPHIGH6",
    "TOLDHI3",
    "CHOLMED3",
    "BPMEDS1",
    "DEAF",
    "BLIND",
    "DECIDE",
    "DIFFWALK",
    "DIFFDRES",
    "DIFFALON",
    "FLUSHOT7",
    "PNEUVAC4",
    "RENTHOM1",
    "DIABAGE4",
    "PDIABTS1",
    "PREDIAB2",
    "DIABTYPE",
    "INSULIN1",
    "CHKHEMO3",
    "EYEEXAM1",
    "DIABEYE1",
    "DIABEDU1",
    "FEETSORE",
]

GENERAL_RECODE_9_ONLY_COLUMNS = [
    "MARITAL",
    "EDUCA",
    "EMPLOY1",
    "_SEX",
    "_AGE65YR",
    "_AGE_G",
    "_CHLDCNT",
    "_EDUCAG",
    "_INCOMG1",
    "_BMI5CAT",
    "_RFBMI5",
    "_RFHLTH",
    "_PHYS14D",
    "_MENT14D",
    "_HLTHPL2",
    "_HCVU654",
    "_TOTINDA",
    "_SMOKER3",
    "_RFSMOK3",
    "_RFDRHV9",
    "_RFBING6",
    "_RFHYPE6",
    "_RFCHOL3",
    "_MICHD",
    "_DRDXAR2",
    "_FLSHOT7",
    "_PNEUMO3",
    "_RACE",
    "_RACEGR3",
    "_HISPANC",
    "_MRACE1",
]

TRAIN_SCHEMA_PATH = PROCESSED_DIR / "train_schema.csv"

ALL_DIRS = [
    DATA_DIR,
    RAW_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    LOGS_DIR,
    OUTPUTS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
]
