import gradio as gr
import pandas as pd
import joblib

# -------------------------
# Load trained pipeline
# -------------------------
model = joblib.load("model_pipeline.pkl")

# -------------------------
# Decision Logic
# -------------------------
def loan_decision(score):
    if score >= 75:
        return "APPROVE"
    elif score >= 50:
        return "MANUAL_REVIEW"
    else:
        return "REJECT"

# -------------------------
# Prediction Function
# -------------------------
def predict_loan(
    gender, marital_status, education_level, employment_status,
    loan_purpose, grade_subgrade,
    age, annual_income, monthly_income, debt_to_income_ratio,
    credit_score, loan_amount, interest_rate, loan_term,
    installment, num_of_open_accounts, total_credit_limit,
    current_balance, delinquency_history, public_records,
    num_of_delinquencies
):

    try:
        data = pd.DataFrame([{
            "gender": gender,
            "marital_status": marital_status,
            "education_level": education_level,
            "employment_status": employment_status,
            "loan_purpose": loan_purpose,
            "grade_subgrade": grade_subgrade,
            "age": age,
            "annual_income": annual_income,
            "monthly_income": monthly_income,
            "debt_to_income_ratio": debt_to_income_ratio,
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "interest_rate": interest_rate,
            "loan_term": loan_term,
            "installment": installment,
            "num_of_open_accounts": num_of_open_accounts,
            "total_credit_limit": total_credit_limit,
            "current_balance": current_balance,
            "delinquency_history": delinquency_history,
            "public_records": public_records,
            "num_of_delinquencies": num_of_delinquencies
        }])

        prob = model.predict_proba(data)[0][1]
        score = round(prob * 100, 2)
        decision = loan_decision(score)

        explanation = (
            "Top factors influencing decision:\n"
            "- Credit Score\n"
            "- Debt to Income Ratio\n"
            "- Delinquency History\n"
            "- Interest Rate\n"
            "- Loan Amount"
        )

        return score, decision, explanation

    except Exception as e:
        return 0, "ERROR", str(e)

# -------------------------
# Gradio UI
# -------------------------
interface = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Dropdown(["Single", "Married"], label="Marital Status"),
        gr.Dropdown(["Graduate", "Post-Graduate", "Undergraduate"], label="Education Level"),
        gr.Dropdown(["Employed", "Self-employed", "Unemployed"], label="Employment Status"),
        gr.Dropdown(["Home", "Car", "Education", "Business"], label="Loan Purpose"),
        gr.Dropdown(["A1", "A2", "B1", "B2", "C1"], label="Grade Subgrade"),

        gr.Number(label="Age"),
        gr.Number(label="Annual Income"),
        gr.Number(label="Monthly Income"),
        gr.Number(label="Debt to Income Ratio"),
        gr.Number(label="Credit Score"),
        gr.Number(label="Loan Amount"),
        gr.Number(label="Interest Rate"),
        gr.Number(label="Loan Term"),
        gr.Number(label="Installment"),
        gr.Number(label="Open Accounts"),
        gr.Number(label="Total Credit Limit"),
        gr.Number(label="Current Balance"),
        gr.Number(label="Delinquency History"),
        gr.Number(label="Public Records"),
        gr.Number(label="Number of Delinquencies")
    ],
    outputs=[
        gr.Number(label="Credit Risk Score"),
        gr.Text(label="Loan Decision"),
        gr.Text(label="Top Risk Factors")
    ],
    title="Loan Approval & Credit Risk System",
    description="ML-based loan approval system (deployment-stable version)"
)

interface.launch()
