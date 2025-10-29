from fastapi import Body, FastAPI

from .utils import read_csv_from_s3

app = FastAPI(title="Health data Demo")

@app.get("/hello")
def read_root():
    return {"msg": "Hello AWS!"}

@app.post("/avg")
def average(data: dict = Body(...)):
    values = [1, 2, 3, 4, data.get("value", 0)]
    mean = sum(values) / len(values)
    return {"mean": mean}

@app.get("/s3demo")
def s3_demo():
    df = read_csv_from_s3("health-demo-hherlangen", "data/labs.csv")
    print(df.head())  # helps confirm what’s loaded
    return {
        "head": df.head(5).to_dict(orient="records")
    }