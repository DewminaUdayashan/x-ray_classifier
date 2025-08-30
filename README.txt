#Run API
uvicorn main:app --reload
* `main`: Refers to the `main.py` file.
* `app`: Refers to the `app = FastAPI()` object inside `main.py`.
* `--reload`: Makes the server restart automatically after you change the code.