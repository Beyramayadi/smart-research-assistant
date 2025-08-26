.PHONY: venv install api ui clean

venv:
\tpython -m venv .venv
\t. .venv/bin/activate && python -m pip install --upgrade pip

install:
\t. .venv/bin/activate && pip install -r requirements.txt

api:
\t. .venv/bin/activate && uvicorn app.main:app --reload --port 8000

ui:
\t. .venv/bin/activate && streamlit run ui/app.py

clean:
\trm -rf .chroma .cache __pycache__
