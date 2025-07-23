import os
from dotenv import load_dotenv # Import the function# type: ignore
from langchain_core.prompts import PromptTemplate# type: ignore
from langchain_openai import ChatOpenAI# type: ignore

# Load environment variables from the .env file
load_dotenv()

# The rest of your code remains exactly the same.
# The `os.getenv("OPENROUTER_API_KEY")` call will now automatically
# find the key that was loaded from your .env file.

class Agent:
    def __init__(self, medical_report=None, role=None, extra_info=None):
        self.medical_report = medical_report
        self.role = role
        self.extra_info = extra_info
        
        # This initialization now works with the key from your .env file
        self.model = ChatOpenAI(
            model="mistralai/mistral-7b-instruct:free",
            temperature=0,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"), # This now reads from the loaded .env
            
        )

        # Initialize the prompt (your logic here is unchanged)
        self.prompt_template = self.create_prompt_template()

    def create_prompt_template(self):
        # This entire method remains the same as your original code
        if self.role == "MultidisciplinaryTeam":
            templates = f"""
                You are a multidisciplinary healthcare team (Cardiologist, Psychologist, Pulmonologist) reviewing patient Michael Johnson's combined reports.

Your task:

1. **Briefly summarize** the patient's overall health status based on the merged insights from all specialties.
2. **Identify 3 specific health concerns**, rooted in the findings. For each, briefly justify why it matters.
3. **Propose a concise treatment plan** that:
   - Integrates input from all specialties,
   - Is personalized to Michael Johnson's symptoms,
   - Is divided into **short-term actions (0–1 month)** and **long-term strategies (3–12 months)**,
   - Covers key aspects: lifestyle, medication, therapy, monitoring,
   - Justifies each recommendation clearly but briefly.

**Guidelines**:
- Prioritize clarity and **conciseness** (max 512 tokens allowed).
- Avoid repeating the original report texts.
- Do not list individual reports — merge insights into a single narrative.
- Use clinical, compassionate language. Stay focused and avoid generic advice.

                Cardiologist Report: {self.extra_info.get('cardiologist_report', '')}
                Psychologist Report: {self.extra_info.get('psychologist_report', '')}
                Pulmonologist Report: {self.extra_info.get('pulmonologist_report', '')}
            """
        else:
            # Note: The placeholder here should be {self.medical_report}, not {medical_report}
            # Since you format it later with self.medical_report, I'm keeping your original structure.
            templates = {
                "Cardiologist": f"""
                    Act like a cardiologist. You will receive a medical report of a patient.
                    Task: Review the patient's cardiac workup, including ECG, blood tests, Holter monitor results, and echocardiogram.
                    Focus: Determine if there are any subtle signs of cardiac issues that could explain the patient’s symptoms. Rule out any underlying heart conditions, such as arrhythmias or structural abnormalities, that might be missed on routine testing.
                    Recommendation: Provide guidance on any further cardiac testing or monitoring needed to ensure there are no hidden heart-related concerns. Suggest potential management strategies if a cardiac issue is identified.
                    Please only return the possible causes of the patient's symptoms and the recommended next steps.
                    Medical Report: {{medical_report}} 
                """,
                "Psychologist": f"""
                    Act like a psychologist. You will receive a patient's report.
                    Task: Review the patient's report and provide a psychological assessment.
                    Focus: Identify any potential mental health issues, such as anxiety, depression, or trauma, that may be affecting the patient's well-being.
                    Recommendation: Offer guidance on how to address these mental health concerns, including therapy, counseling, or other interventions.
                    Please only return the possible mental health issues and the recommended next steps.
                    Patient's Report: {{medical_report}}
                """,
                "Pulmonologist": f"""
                    Act like a pulmonologist. You will receive a patient's report.
                    Task: Review the patient's report and provide a pulmonary assessment.
                    Focus: Identify any potential respiratory issues, such as asthma, COPD, or lung infections, that may be affecting the patient's breathing.
                    Recommendation: Offer guidance on how to address these respiratory concerns, including pulmonary function tests, imaging studies, or other interventions.
                    Please only return the possible respiratory issues and the recommended next steps.
                    Patient's Report: {{medical_report}}
                """
            }
            templates = templates[self.role]
        # Using f-string for the dictionary requires an extra step for the PromptTemplate
        return PromptTemplate.from_template(templates)
    
    def run(self):
        print(f"{self.role} is running...")
        # The MultidisciplinaryTeam prompt doesn't have a 'medical_report' variable, so we handle that
        if self.role == "MultidisciplinaryTeam":
            prompt_input = self.prompt_template.format()
        else:
            prompt_input = self.prompt_template.format(medical_report=self.medical_report)
        
        try:
            response = self.model.invoke(prompt_input)
            return response.content
        except Exception as e:
            print("Error occurred:", e)
            return None

# The rest of your specialized agent classes remain unchanged
class Cardiologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Cardiologist")

class Psychologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Psychologist")

class Pulmonologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Pulmonologist")

class MultidisciplinaryTeam(Agent):
    def __init__(self, cardiologist_report, psychologist_report, pulmonologist_report):
        extra_info = {
            "cardiologist_report": cardiologist_report,
            "psychologist_report": psychologist_report,
            "pulmonologist_report": pulmonologist_report
        }
        super().__init__(role="MultidisciplinaryTeam", extra_info=extra_info)