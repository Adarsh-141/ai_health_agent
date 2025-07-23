# app.py

import streamlit as st
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Import your custom agent classes
from Utils.Agents import Cardiologist, Psychologist, Pulmonologist, MultidisciplinaryTeam

# --- Load Environment Variables ---
load_dotenv()

# --- Email Sending Function ---
def send_email(recipient_email, subject, body):
    """Sends an email using credentials from .env file."""
    sender_email = os.getenv("SENDER_EMAIL")
    password = os.getenv("SENDER_PASSWORD")

    if not all([sender_email, password]):
        st.error("Email credentials are not set in the .env file. Cannot send email.")
        return False

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = recipient_email

    # Attach the body to the email
    message.attach(MIMEText(body, "plain"))

    try:
        # Create a secure SSL context
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, recipient_email, message.as_string())
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# --- Agent Analysis Function ---
def run_full_analysis(medical_report):
    """
    Runs the full multi-agent analysis and returns the final diagnosis.
    This function encapsulates the logic from your original main.py.
    """
    agents = {
        "Cardiologist": Cardiologist(medical_report),
        "Psychologist": Psychologist(medical_report),
        "Pulmonologist": Pulmonologist(medical_report)
    }
    
    responses = {}
    with ThreadPoolExecutor(max_workers=len(agents)) as executor:
        future_to_agent = {executor.submit(agent.run): name for name, agent in agents.items()}
        for future in as_completed(future_to_agent):
            agent_name = future_to_agent[future]
            try:
                responses[agent_name] = future.result()
            except Exception as e:
                responses[agent_name] = f"Error during {agent_name} analysis: {e}"

    team_agent = MultidisciplinaryTeam(
        cardiologist_report=responses.get("Cardiologist", "N/A"),
        psychologist_report=responses.get("Psychologist", "N/A"),
        pulmonologist_report=responses.get("Pulmonologist", "N/A")
    )

    final_diagnosis = team_agent.run()
    return final_diagnosis

# --- Streamlit User Interface ---
st.set_page_config(page_title="Medical AI Analysis", layout="wide")

st.title("🩺 Medical Report Analysis AI")
st.write("Paste a patient's medical report below. The AI agents (Cardiologist, Psychologist, and Pulmonologist) will analyze it, and a multidisciplinary team will provide a consolidated summary and action plan.")

# Initialize session state to hold the analysis result
if 'final_diagnosis' not in st.session_state:
    st.session_state.final_diagnosis = None

# Input text area for the medical report
medical_report_input = st.text_area(
    "Paste the patient's medical report here:",
    height=300,
    placeholder="e.g., Patient Name: Adarsh, Age: 20..."
)

# Analyze button
if st.button("Analyze Report", type="primary"):
    if medical_report_input.strip():
        with st.spinner("🧠 The specialist AI agents are analyzing the report... This may take a moment."):
            try:
                # Run the analysis and store the result in session state
                st.session_state.final_diagnosis = run_full_analysis(medical_report_input)
                st.success("Analysis complete!")
            except Exception as e:
                st.error(f"An unexpected error occurred during analysis: {e}")
                st.session_state.final_diagnosis = None
    else:
        st.warning("Please paste a medical report before analyzing.")

# --- Display the result and email options ---
if st.session_state.final_diagnosis:
    st.markdown("---")
    st.subheader("Multidisciplinary Team Analysis")
    st.markdown(st.session_state.final_diagnosis)

    # Save to file locally (optional)
    try:
        output_path = os.path.join("results", "final_diagnosis.txt")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(st.session_state.final_diagnosis)
        st.info(f"Report has also been saved to `{output_path}`")
    except Exception as e:
        st.error(f"Could not save file locally: {e}")


    st.markdown("---")
    st.subheader("📧 Email Report")
    
    email_option = st.radio(
        "Would you like to email this report?",
        ("No", "Yes"),
        horizontal=True,
        index=0 # Default to "No"
    )

    if email_option == "Yes":
        recipient_email = st.text_input("Enter recipient's email address:")
        if st.button("Send Email"):
            if recipient_email:
                with st.spinner("Sending email..."):
                    subject = "AI-Generated Medical Report Analysis"
                    body = f"Here is the AI-generated analysis based on the provided medical report:\n\n---\n\n{st.session_state.final_diagnosis}"
                    
                    if send_email(recipient_email, subject, body):
                        st.success(f"Email successfully sent to {recipient_email}!")
                    # The send_email function will show an st.error on failure
            else:
                st.warning("Please enter a recipient email address.")