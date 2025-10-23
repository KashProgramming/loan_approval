# Loan Approval Prediction

A Machine Learning web app that predicts whether a loan application should be **approved or rejected** based on applicant financial, personal, and credit history details.

🚀 **Live Demo**: [Loan Approval Prediction](https://loan-approval-m9xu.onrender.com)
📌 **Tech Stack**: Python · Flask · Scikit-learn · HTML/CSS
🧠 **Model Used**: Logistic Regression (best-performing classifier with a test accuracy of 99.9857%)

---

## Project Overview
Financial institutions rely on multiple risk indicators like credit score, employment type, DTI ratio, and repayment history before approving a loan.
This project builds an end-to-end ML pipeline to **predict loan approval decisions automatically**.

The project includes:
* Data preprocessing & feature engineering
* Scaling of numerical features
* Encoding categorical variables
* Model selection & training
* Model serialization (`.pkl`)
* Flask-based deployment
* Frontend form for predictions
* Live cloud deployment on Render

---

## ML Pipeline

| Step                | Description                                            |
| ------------------- | ------------------------------------------------------ |
| Data Cleaning       | Handled missing values + formatted features            |
| Feature Engineering | Created ratios, processed credit history duration etc. |
| Scaling             | StandardScaler used for numeric columns                |
| Model               | Logistic Regression trained + evaluated                |
| Saving Model        | Exported using `pickle`                                |
| Deployment          | Flask API + HTML frontend                              |

---

## Project Structure
```
loan-approval/
│
├── app.py                 # Flask backend
├── model.pkl              # Trained ML model
├── scaler.pkl             # Fitted scaler
├── templates/
│   └── home.html         # Frontend
├── static/
│   └── (optional css/js)
└── README.md
```

---

## Run Locally
```bash
# clone repo
git clone <repo-url>
cd loan-approval

# create env
python3 -m venv venv
source venv/bin/activate   # mac/linux
# venv\Scripts\activate    # windows

# install requirements
pip install -r requirements.txt

# run flask app
python app.py
```
Open `http://127.0.0.1:5000` in your browser ✅

---

## Deployment
The app is live on **Render** and publicly accessible.
You can also redeploy anytime by connecting GitHub → Render with automatic builds.

---

## Future Improvements
* Add XGBoost / Ensemble comparison
* User authentication / Admin dashboard
* Model explainability with SHAP
* Better UI (React / Tailwind)
* Cloud database for logging predictions
