
from pathlib import Path

BASE_DIR = Path("/mnt/c/Users/sinfo/Desktop/pytorch/django/drf/appv1")
VOCAB_FILE = BASE_DIR / "vocab/vocab.txt"
BERT_CONFIG = BASE_DIR / "weights/bert_config.json"
model_file = BASE_DIR / "weights/pytorch_model.bin"
MODEL_FILE = BASE_DIR / "data/bert_fine_tuning_chABSA.pth"
PKL_FILE = BASE_DIR / "data/text.pkl"
DATA_PATH = BASE_DIR / "data"
max_length = 256
