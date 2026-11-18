import gradio as gr
import joblib

model = joblib.load("loan_model.pkl")

def predict_loan(age, monthly_income, credit_score, loan_amount):
    data = [[age, monthly_income, credit_score, loan_amount]]

    prediction = model.predict(data)[0]
    probabilities = model.predict_proba(data)[0]
    confidence = max(probabilities) * 100

    if prediction == 1:
      
        return (
            "Loan Prediction Result\n\n"
            "Status: APPROVED\n\n"
            "The applicant is likely to be eligible "
            "based on the information provided."
        )
    else:
        return (
            "Loan Prediction Result\n\n"
            "Status: REJECTED\n\n"
            "The applicant is currently unlikely to be eligible "
            "based on the information provided."
        )
demo = gr.Interface(
    fn=predict_loan,
   
inputs=[
    gr.Slider(18, 60, value=30, step=1, label="Age"),

    gr.Number(
        value=50000,
        minimum=20000,
        maximum=200000,
        label="Monthly Income (₹)"
    ),

    gr.Slider(300, 850, value=700, step=1, label="Credit Score"),

    gr.Number(
        value=1500000,
        minimum=50000,
        maximum=5000000,
        label="Loan Amount (₹)"
    )
],

    outputs=gr.Textbox(label="Prediction Result"),
    title="Loan Approval Prediction System",
    description="Enter applicant details to predict loan approval.",
    flagging_mode="never",
)

demo.launch()