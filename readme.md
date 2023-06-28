## :blue_book: :book: :orange_book: :neckbeard: Books recommendation project :neckbeard: :orange_book: :book: :blue_book:

**Project description:**
We are building the recommendation system for books.

Project install steps:
- clone repo
`git clone https://github.com/naz2001r/recsys_project_MAUN.git`
- create virtual environment
`python -m venv env`
- activate virtual environment `source env/bin/activate` (for Linux and MacOS)
- install dependecies using pip
`pip install -r requirements.txt`
- pull needed data using dvc (it can take up to 20 minutes, please be patient, to see the progress we use verbose flag in here)
`dvc pull -v`
- reproduce training pipeline
`dvc repro`
- to run single step of pipelint
`dvc repro -sf STEPNAME`

[DVC documentation](https://dvc.org/doc/start/data-management/data-versioning)

For more information reach us via slack