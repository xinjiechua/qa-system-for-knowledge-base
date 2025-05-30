import logging
from src.gradio import launch_ui
from src.vector import VectorDB

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

CLEAR_DATA = False     # Clear all data from the vector database
INSERT_DATA = False   # Parse and insert all data into the vector database

if __name__ == "__main__":
    if CLEAR_DATA:
        VectorDB().clear()
        
    if INSERT_DATA:
        VectorDB().insert()

    launch_ui()