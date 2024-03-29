from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import os
import pandas as pd

app = FastAPI()


@app.post("/csv")
async def dummy(file: UploadFile = File(...)):
    # Path to the CSV file in the project folder
    csv_file_path = "./DailyDelhiClimateTest.csv"

    # Check if the file exists
    if not os.path.exists(csv_file_path):
        return {"error": "CSV file not found"}

    df = pd.read_csv(csv_file_path)

    return df.to_dict("dict") 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)