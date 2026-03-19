import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4