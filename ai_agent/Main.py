# Importing the needed modules 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv # type: ignore
# Assuming your Agent classes are in Utils/Agents.py
from Utils.Agents import Cardiologist, Psychologist, Pulmonologist, MultidisciplinaryTeam
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Loading API key from the .env file in the root directory.
load_dotenv()

# --- Read the medical report using an OS-independent path ---
try:
    report_path = os.path.join("Medical Reports", "Medical Report - Michael Johnson - Panic Attack Disorder.txt")
    with open(report_path, "r") as file:
        medical_report = file.read()
except FileNotFoundError:
    print(f"Error: The medical report file was not found at {report_path}")
    exit() # Exit the script if the report is not found

# --- Initialize the specialist agents ---
agents = {
    "Cardiologist": Cardiologist(medical_report),
    "Psychologist": Psychologist(medical_report),
    "Pulmonologist": Pulmonologist(medical_report)
}

# --- Function to run each agent and get their response ---
def get_response(agent_name, agent):
    """Runs an agent and returns its name and response."""
    print(f"Running {agent_name} analysis...")
    response = agent.run()
    return agent_name, response

# --- Run agents concurrently and collect responses with robust error handling ---
responses = {}
with ThreadPoolExecutor(max_workers=len(agents)) as executor:
    futures = {executor.submit(get_response, name, agent): name for name, agent in agents.items()}
    
    for future in as_completed(futures):
        agent_name_from_future = futures[future]
        try:
            agent_name, response = future.result()
            responses[agent_name] = response
            print(f"Successfully received analysis from {agent_name}.")
        except Exception as e:
            print(f"An error occurred while running the {agent_name_from_future} agent: {e}")
            responses[agent_name_from_future] = "Error: Analysis could not be completed."

# --- Run the MultidisciplinaryTeam agent ---
team_agent = MultidisciplinaryTeam(
    cardiologist_report=responses.get("Cardiologist", "N/A"),
    psychologist_report=responses.get("Psychologist", "N/A"),
    pulmonologist_report=responses.get("Pulmonologist", "N/A")
)

print("\nRunning Multidisciplinary Team analysis...")
final_diagnosis = team_agent.run()

# --- Save the final diagnosis to a file (with a final check) ---
txt_output_path = os.path.join("results", "final_diagnosis.txt")
os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)

if final_diagnosis: # Check if the final diagnosis is not None
    final_diagnosis_text = "### Final Diagnosis:\n\n" + final_diagnosis
    with open(txt_output_path, "w") as txt_file:
        txt_file.write(final_diagnosis_text)
    print(f"\n✅ Final diagnosis has been saved to {txt_output_path}")
else:
    error_message = "The Multidisciplinary Team analysis failed and could not generate a final report."
    with open(txt_output_path, "w") as txt_file:
        txt_file.write(error_message)
    print(f"\n❌ {error_message}")