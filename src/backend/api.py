from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import os

app = FastAPI()


@app.post("/csv")
async def dummy(file: UploadFile = File(...)):
    # Path to the CSV file in the project folder
    csv_file_path = "./DailyDelhiClimateTest.csv"

    # Check if the file exists
    if not os.path.exists(csv_file_path):
        return {"error": "CSV file not found"}

    # Open the CSV file and return as a streaming response
    with open(csv_file_path, mode='rb') as csv_file:
        csv_content = csv_file.read()

    return StreamingResponse(iter([csv_content]), media_type="text/csv", headers={"Content-Disposition": "attachment;filename=data.csv"}) 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)