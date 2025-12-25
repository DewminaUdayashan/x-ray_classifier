#Run API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
uvicorn main:app --reload
* `main`: Refers to the `main.py` file.
* `app`: Refers to the `app = FastAPI()` object inside `main.py`.
* `--reload`: Makes the server restart automatically after you change the code.