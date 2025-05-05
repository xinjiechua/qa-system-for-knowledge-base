from src.gradio import launch_ui
from src.vector import VectorDB
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    VectorDB().insert()
    launch_ui()