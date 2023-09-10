# #importing the libraries;
# from fastapi import FastAPI
# from uvicorn import run as uvicorn_run
# from typing import List, Literal
# from pydantic import BaseModel
# import joblib
# from fastapi.encoders import jsonable_encoder
# import uvicorn


# #CONFIG
# app = FastAPI(
#     title="Income Prediction API",
#     version="0.0.1",
#     description="This is a simple API for Income prediction")

# #API INPUT
# class Input(BaseModel):
#     age: int 
#     gender: object 
#     education: object 
#     marital_status: object 
#     race: object 
#     employment_stat: int
#     wage_per_hour: int
#     working_week_per_year: int  
#     industry_code: int
#     occupation_code: int  
#     total_employed: int 
#     vet_benefit: int  
#     tax_status: object 
#     gains: int 
#     losses: int  
#     stocks_status: int 
#     citizenship: object 
#     mig_year: int  
#     importance_of_record: float
#     #income_above_limit
    
    
#     #ENDPOINT
#     @app.get("/")
#     async def root():
#         return {"message": "Online"}
    
#     # @app.post("/predict")
#     # def predict(input: Input):
#     #     scaler = joblib.load("Assets/scaler.joblib")
#     #     model = joblib.load("Assets/model.joblib")
        
#     #     features = [input.age,    
#     # input.gender,                 
#     # input.education,                 
#     # input.marital_status,               
#     # input.race,                     
#     # input.employment_stat,
#     # input.wage_per_hour,
#     # input.working_week_per_year,
#     # input.industry_code,
#     # input.occupation_code,
#     # input.total_employed,
#     # input.vet_benefit,
#     # input.tax_status,
#     # input.gains,
#     # input.losses,
#     # input.stocks_status,
#     # input.citizenship,
#     # input.mig_year,
#     # input.importance_of_record]
    
    
#     # scale_features = scaler.transform([features])[0]
#     # prediction = model.predict([scale_features])[0]
    
#     # #Serializing the prediction using jsonable_encoder
#     # serialized_prediction = jsonable_encoder({"prediction": int(prediction)})
#     # if serialized_prediction == 1:
#     #     result=  {"prediction": "Income above limit"}
#     # else:
#     #     result= {"prediction": "Income below limit"}
#     #     return result
        
# if __name__ == "__main__":
#     uvicorn.run('app:app', reload =True)
#         #uvicorn.run(app, host="000000000", port=8000)
        

# # Importing the libraries
# from fastapi import FastAPI
# from uvicorn import run as uvicorn_run
# from typing import List, Literal
# from pydantic import BaseModel
# import joblib
# from fastapi.encoders import jsonable_encoder
# import uvicorn

# # CONFIG
# app = FastAPI(
#     title="Income Prediction API",
#     version="0.0.1",
#     description="This is a simple API for Income prediction")

# # API INPUT
# class Input(BaseModel):
#     age: int 
#     gender: object 
#     education: object 
#     marital_status: object 
#     race: object 
#     employment_stat: int
#     wage_per_hour: int
#     working_week_per_year: int  
#     industry_code: int
#     occupation_code: int  
#     total_employed: int 
#     vet_benefit: int  
#     tax_status: object 
#     gains: int 
#     losses: int  
#     stocks_status: int 
#     citizenship: object 
#     mig_year: int  
#     importance_of_record: float

# # ENDPOINT
# @app.get("/")
# async def root():
#     return {"message": "Online"}

# @app.post("/predict")
# def predict(input: Input):
#     encoder = joblib.load("Assets/encoder.joblib")
#     scaler = joblib.load("Assets/scaler.joblib")
#     model = joblib.load("Assets/model.joblib")
        
#     features = [input.age,    
#     input.gender,                 
#     input.education,                 
#     input.marital_status,               
#     input.race,                     
#     input.employment_stat,
#     input.wage_per_hour,
#     input.working_week_per_year,
#     input.industry_code,
#     input.occupation_code,
#     input.total_employed,
#     input.vet_benefit,
#     input.tax_status,
#     input.gains,
#     input.losses,
#     input.stocks_status,
#     input.citizenship,
#     input.mig_year,
#     input.importance_of_record]
    
#     encode_features = encoder.transform([features])[0]
#     scale_features = scaler.transform([encode_features])[0]
#     prediction = model.predict([scale_features])[0]
    
#     # Serializing the prediction using jsonable_encoder
#     serialized_prediction = jsonable_encoder({"prediction": int(prediction)})
#     if serialized_prediction == 1:
#         result = {"prediction": "Income above limit"}
#     else:
#         result = {"prediction": "Income below limit"}
#     return result

# if __name__ == "__main__":
#     uvicorn.run('app:app', reload=True)

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import joblib
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="Income Prediction API",
    version="0.0.1",
    description="This is a simple API for Income prediction"
)

# Define input model
class Input(BaseModel):
    age: int
    gender: str  # Change 'object' to 'str' if gender is a string
    education: str
    marital_status: str
    race: str
    employment_stat: int
    wage_per_hour: int
    working_week_per_year: int
    industry_code: int
    occupation_code: int
    total_employed: int
    vet_benefit: int
    tax_status: str
    gains: int
    losses: int
    stocks_status: int
    citizenship: str
    mig_year: int
    importance_of_record: float

# Define prediction endpoint
@app.post("/predict")
def predict(input: Input):
    # Load your model and encoders
    encoder = joblib.load("Assets/encoder.joblib")
    scaler = joblib.load("Assets/scaler.joblib")
    model = joblib.load("Assets/model.joblib")
        
    # Prepare input features
    features = [input.age,    
                input.gender,                 
                input.education,                 
                input.marital_status,               
                input.race,                     
                input.employment_stat,
                input.wage_per_hour,
                input.working_week_per_year,
                input.industry_code,
                input.occupation_code,
                input.total_employed,
                input.vet_benefit,
                input.tax_status,
                input.gains,
                input.losses,
                input.stocks_status,
                input.citizenship,
                input.mig_year,
                input.importance_of_record]
    
    # Encode and scale features
    encode_features = encoder.transform([features])[0]
    scale_features = scaler.transform([encode_features])[0]
    
    # Make a prediction
    prediction = model.predict([scale_features])[0]
    
    # Determine the prediction result
    if prediction == 1:
        result = {"prediction": "Income above limit"}
    else:
        result = {"prediction": "Income below limit"}
    
    return result

# Define root endpoint
@app.get("/")
async def root():
    return {"message": "Online"}

if __name__ == "__main__":
    uvicorn.run('app:app', reload=True)
