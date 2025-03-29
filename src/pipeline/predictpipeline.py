import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
        def __init__(self):
            pass

        def predict(self, features):
            try:
                model_path = os.path.join("artifacts", "model.pkl")
                preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
                print("Before Loading")
                model = load_object(file_path=model_path)
                preprocessor = load_object(file_path=preprocessor_path)
                print("After Loading")
                data_scaled = preprocessor.transform(features)
                preds = model.predict(data_scaled)
                return preds

            except Exception as e:
                raise CustomException(e, sys)


class CustomData:
        def __init__(self, 
        checking_balance: object,
            months_loan_duration: int,
            credit_history: object,
            purpose: object,
            amount: int,
            savings_balance: object,
            employment_duration: object,
            percent_of_income: int,
            years_at_residence: int,
            age: int,
            other_credit: object,
            housing: object,
            existing_loans_count: int,
            job: object,
            dependents: int,
            phone: object):        
            self.checking_balance = checking_balance
            self.months_loan_duration = months_loan_duration
            self.credit_history = credit_history
            self.purpose = purpose
            self.amount = amount
            self.savings_balance = savings_balance
            self.employment_duration = employment_duration
            self.percent_of_income = percent_of_income
            self.years_at_residence = years_at_residence
            self.age = age
            self.other_credit = other_credit
            self.housing = housing
            self.existing_loans_count = existing_loans_count
            self.job = job
            self.dependents = dependents
            self.phone = phone

        def get_data_as_data_frame(self):
            try:
                custom_data_input_dict = {
                    "checking_balance": [self.checking_balance],
                    "months_loan_duration": [self.months_loan_duration],
                    "credit_history": [self.credit_history],
                    "purpose": [self.purpose],
                    "amount": [self.amount],
                    "savings_balance": [self.savings_balance],
                    "employment_duration": [self.employment_duration],
                    "percent_of_income": [self.percent_of_income],
                    "years_at_residence": [self.years_at_residence],
                    "age": [self.age],
                    "other_credit": [self.other_credit],
                    "housing": [self.housing],
                    "existing_loans_count": [self.existing_loans_count],
                    "job": [self.job],
                    "dependents": [self.dependents],
                    "phone": [self.phone]
                }
                return pd.DataFrame(custom_data_input_dict)
            except Exception as e:
                raise CustomException(e, sys)
