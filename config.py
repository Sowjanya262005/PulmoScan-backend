import os
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
EXPLAIN = os.getenv("EXPLAIN", "false").lower() in {"1","true","yes"}
IMG_SIZE = int(os.getenv("IMG_SIZE", "384"))
