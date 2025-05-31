import logging
import asyncio
from src.gradio import launch_ui
from src.vector import VectorDB

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

CLEAR_DATA = False    # Clear all data from the vector database
INSERT_DATA = False   # Parse and insert all data into the vector database

async def main():
    vectordb = VectorDB()
    if CLEAR_DATA:
        vectordb.clear()
    if INSERT_DATA:
        await vectordb.insert()
    launch_ui()

if __name__ == "__main__":
    asyncio.run(main())