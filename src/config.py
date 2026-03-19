"""Recommendation Engine - Configuration."""
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_FACTORS = 100
N_EPOCHS_MF = 20
MIN_USER_RATINGS = 20
MIN_ITEM_RATINGS = 10
TEST_RATIO = 0.2
NCF_EMBEDDING_DIM = 64
NCF_BATCH_SIZE = 1024
NCF_EPOCHS = 20

FEATURE_WEIGHTS = {"genre": 0.4, "description": 0.3, "cast": 0.2, "director": 0.1}
ENSEMBLE_WEIGHTS = {"cf": 0.4, "content": 0.3, "ncf": 0.3}
MMR_DIVERSITY_WEIGHT = 0.3

API_HOST = "0.0.0.0"
API_PORT = 8001

STREAMLIT_THEME = {
    "primaryColor": "#1f77b4",
    "backgroundColor": "#0e1117",
    "secondaryBackgroundColor": "#262730",
    "textColor": "#ffffff",
}
