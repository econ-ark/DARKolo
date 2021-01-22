
# check https://python-poetry.org/docs/
# curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -


poetry install
poetry run python -m ipykernel install --user --name=uptodate

