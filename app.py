import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load model and scaler
lr_model = pickle.load(open("lr_loan_approval.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

# Categorical mappings (must match training)
emp_map = {"Employed":0,"Self-Employed":1,"Unemployed":2}
edu_map = {"Bachelor":0,"High School":1,"Associate":2,"Master":3,"Doctorate":4}
marital_map = {"Married":0,"Single":1,"Divorced":2,"Widowed":3}
home_map = {"Mortgage":0,"Rent":1,"Own":2,"Other":3}
purpose_map = {"Home":0,"Debt Consolidation":1,"Auto":2,"Education":3,"Other":4}

feature_order = [
    "AnnualIncome","CreditScore","EmploymentStatus","EducationLevel","Experience",
    "LoanAmount","LoanDuration","MaritalStatus","NumberOfDependents",
    "HomeOwnershipStatus","MonthlyDebtPayments","CreditCardUtilizationRate",
    "NumberOfOpenCreditLines","NumberOfCreditInquiries","DebtToIncomeRatio",
    "BankruptcyHistory","LoanPurpose","PreviousLoanDefaults","PaymentHistory",
    "LengthOfCreditHistory","SavingsAccountBalance","CheckingAccountBalance",
    "TotalLiabilities","UtilityBillsPaymentHistory","JobTenure","NetWorth",
    "BaseInterestRate","InterestRate","TotalDebtToIncomeRatio","RiskScore"
]

def preprocess_input(data_dict):
    """Convert input dict to scaled numpy array with categorical mappings"""
    input_list = [data_dict[col] for col in feature_order]

    # Apply categorical mappings
    input_list[2] = emp_map[input_list[2]]
    input_list[3] = edu_map[input_list[3]]
    input_list[7] = marital_map[input_list[7]]
    input_list[9] = home_map[input_list[9]]
    input_list[16] = purpose_map[input_list[16]]

    # Convert numeric strings to float
    input_list = [float(x) if isinstance(x,str) and x.replace('.','',1).isdigit() else x for x in input_list]

    # Scale
    input_array = np.array(input_list).reshape(1,-1)
    return scaler.transform(input_array)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json["data"]
    scaled_data = preprocess_input(data)
    output = lr_model.predict(scaled_data)[0]
    return jsonify(int(output))

@app.route("/predict", methods=["POST"])
def predict():
    # Collect data from HTML form
    data = {col: request.form[col] for col in feature_order}
    scaled_data = preprocess_input(data)
    output = lr_model.predict(scaled_data)[0]
    prediction_text = "Loan Approved ✅" if output==0 else "Loan Rejected ❌"
    return render_template("home.html", prediction_text=prediction_text)

if __name__=="__main__":
    app.run(debug=True)
