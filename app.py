from src.gradio import launch_ui
from src.vector import VectorDB
import logging
import warnings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

INSERT_DATA = False

if __name__ == "__main__":
    if INSERT_DATA:
        warnings.warn("Inserting data into the vector database. This may take a while.")
        VectorDB().insert()
    launch_ui()