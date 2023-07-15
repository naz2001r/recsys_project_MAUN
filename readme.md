## :blue_book: :book: :orange_book: :neckbeard: Books recommendation project :neckbeard: :orange_book: :book: :blue_book:

**Project description:**
We are building the recommendation system for books. 

Project install steps:
- clone repo
`git clone https://github.com/naz2001r/recsys_project_MAUN.git`
- create virtual environment
`python -m venv env`
- activate virtual environment `source env/bin/activate` (for Linux and MacOS) | `.\env\Scripts\activate.bat` (for Windows)
We are using poetry to manage dependencies.
- install poetry
`pip install poetry`
- install dependecies using poetry. It could take some time
`poetry install`
- pull needed data using dvc (it can take up to 30 minutes, please be patient, to see the progress we use verbose flag in here)
`dvc pull -v`
- reproduce training pipeline (actually sll steps should be skipped, that will mean that you have latest data locally)
`dvc repro -v`
- to run single step of pipeline
`dvc repro -v -sf STEPNAME`
- to run all steps of pipeline after some step
`dvc repro -v -f STEPNAME --downstream`
- to run all steps of pipeline after some step without running them actually
`dvc repro -v -f STEPNAME --downstream --dry`

[DVC documentation](https://dvc.org/doc/start/data-management/data-versioning)

EDA jupyter notebook is in /data/notebooks folder.

Models developed:
- Baseline model
- Collaborative filtering
- Content-base filtering
- Matrix factorization
- Content-base sentence transformer recomender
- Super-duper hybrid NN recommender

For more information reach us via slack