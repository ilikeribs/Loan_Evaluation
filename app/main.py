import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import json
from pydantic import BaseModel

class loan_dec(BaseModel):

    amount : int
    employment_length : int 
    debt_to_income : float 
    fico : float 

class loan_eval(BaseModel):
    loan_amnt : float 
    term : float
    home_ownership : str
    verification_status : str 
    dti : float  
    total_acc : float
    fico : float
    Year : int 



class loan_sub_grade(BaseModel):
    loan_amnt : float 
    term : float
    home_ownership : str
    grade : str 
    emp_length : float  
    dti : float
    open_acc : float 
    total_acc : float
    fico : float
    Year : int
    CPI : float
    

class loan_int_rate(BaseModel):
    loan_amnt : float 
    term : float
    dti : float
    total_acc : float
    fico : float
    emp_length : float 
    open_acc : float 
    CPI : float 
    exch_rate : float 
    Year : int 
    grade : str
    sub_grade: str 






app = FastAPI()

loan_decision =joblib.load("app/acc_rej_clf.sav")
grade_decision= joblib.load ("app/grade_clf.sav")
sub_grade_clf=joblib.load("app/sub_grade_clf.sav")
int_rate_pred=joblib.load("app/int_rate_pred.sav")


class loanJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, loan_dec):
            return obj.model_dump()
        return super().default(obj)

@app.get('/')
def index():
    return {'message': "Welcome to loan evaluation"}

@app.post('/predict_loan_status', response_class=JSONResponse)
def predict_loan_status(payload: loan_dec):
    try:
        df = pd.DataFrame([payload.model_dump()])
        pred = loan_decision.predict(df)
        result = {"Loan status": pred.tolist()[0]}  
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/predict_loan_grade', response_class=JSONResponse)
def predict_loan_grade(payload: loan_eval):
    try:
        df = pd.DataFrame([payload.model_dump()])
        pred = grade_decision.predict(df)
        result = {"Loan grade": pred.tolist()[0]}  
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/predict_loan_subgrade', response_class=JSONResponse)
def predict_loan_subgrade(payload: loan_sub_grade):
    try:
        df = pd.DataFrame([payload.model_dump()])
        pred = sub_grade_clf.predict(df)
        result = {"Loan sub grade": pred.tolist()[0]}  
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/predict_loan_intrate', response_class=JSONResponse)
def predict_loan_grade(payload: loan_int_rate):
    try:
        df = pd.DataFrame([payload.model_dump()])
        pred = int_rate_pred.predict(df)
        result = {"Loan int rate": pred.tolist()[0]}  
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
