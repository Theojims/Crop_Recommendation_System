import os
import logging
from pathlib import Path

logging.basicConfig(level = logging.INFO, format = "[%(asctime)s]: %(message)s:")

project_name = "CropRecommendation"

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/pipeline/__init__.py",
    "requirements.txt",
    "setup.py",
    "templates/index.html",
    "statics"
]



for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir, exist_ok =True)

        logging.info(f"Creating directory; {filedir} for the file: {filename}")


    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):

        with open(filepath, "w") as file:
            pass
            
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f'{filename} is already existing')