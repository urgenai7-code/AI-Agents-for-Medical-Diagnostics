from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

class Agent:
    def __init__(self, medical_report=None, role=None, extra_info=None):
        self.medical_report = medical_report
        self.role = role
        self.extra_info = extra_info or {}

        # Initialize the prompt based on role and other info
        self.prompt_template = self.create_prompt_template()

        # Initialize the model (expects OPENAI_API_KEY in env or apikey.env handling elsewhere)
        self.model = ChatOpenAI(temperature=0, model="gpt-4o")

    def create_prompt_template(self):
        # MultidisciplinaryTeam uses a single string template (not a dict)
        if self.role == "MultidisciplinaryTeam":
            template_str = (
                "Act like a multidisciplinary team of healthcare professionals.\n"
                "You will receive a medical report of a patient visited by a Cardiologist, Psychologist, and Pulmonologist.\n"
                "Task: Review the patient's medical report from the Cardiologist, Psychologist, and Pulmonologist, analyze them and come up with a list of 3 possible health issues of the patient.\n"
                "Just return a list of bullet points of 3 possible health issues of the patient and for each issue provide the reason.\n\n"
                "Cardiologist Report: {cardiologist_report}\n"
                "Psychologist Report: {psychologist_report}\n"
                "Pulmonologist Report: {pulmonologist_report}\n"
            )
            return PromptTemplate.from_template(template_str)

        # Otherwise, pick from the dict of templates keyed by role
        templates = {
            "Cardiologist": (
                "Act like a cardiologist. You will receive a medical report of a patient.\n"
                "Task: Review the patient's cardiac workup, including ECG, blood tests, Holter monitor results, and echocardiogram.\n"
                "Focus: Determine if there are any subtle signs of cardiac issues that could explain the patient’s symptoms. "
                "Rule out any underlying heart conditions, such as arrhythmias or structural abnormalities, that might be missed on routine testing.\n"
                "Recommendation: Provide guidance on any further cardiac testing or monitoring needed to ensure there are no hidden heart-related concerns. "
                "Suggest potential management strategies if a cardiac issue is identified.\n"
                "Please only return the possible causes of the patient's symptoms and the recommended next steps.\n"
                "Medical Report: {medical_report}\n"
            ),
            "Psychologist": (
                "Act like a psychologist. You will receive a patient's report.\n"
                "Task: Review the patient's report and provide a psychological assessment.\n"
                "Focus: Identify any potential mental health issues, such as anxiety, depression, or trauma, that may be affecting the patient's well-being.\n"
                "Recommendation: Offer guidance on how to address these mental health concerns, including therapy, counseling, or other interventions.\n"
                "Please only return the possible mental health issues and the recommended next steps.\n"
                "Patient's Report: {medical_report}\n"
            ),
            "Pulmonologist": (
                "Act like a pulmonologist. You will receive a patient's report.\n"
                "Task: Review the patient's report and provide a pulmonary assessment.\n"
                "Focus: Identify any potential respiratory issues, such as asthma, COPD, or lung infections, that may be affecting the patient's breathing.\n"
                "Recommendation: Offer guidance on how to address these respiratory concerns, including pulmonary function tests, imaging studies, or other interventions.\n"
                "Please only return the possible respiratory issues and the recommended next steps.\n"
                "Patient's Report: {medical_report}\n"
            ),
        }

        if self.role not in templates:
            raise KeyError(
                f"Unknown role '{self.role}'. Expected one of: {list(templates.keys()) + ['MultidisciplinaryTeam']}"
            )

        return PromptTemplate.from_template(templates[self.role])
    
    def run(self):
        print(f"{self.role} is running...")
        try:
            if self.role == "MultidisciplinaryTeam":
                # MDT uses the three specialist reports (in extra_info), not the single medical_report
                prompt = self.prompt_template.format(
                    cardiologist_report=self.extra_info.get("cardiologist_report", ""),
                    psychologist_report=self.extra_info.get("psychologist_report", ""),
                    pulmonologist_report=self.extra_info.get("pulmonologist_report", "")
                )
            else:
                # Specialist roles use {medical_report}
                prompt = self.prompt_template.format(medical_report=self.medical_report)

            response = self.model.invoke(prompt)
            # ChatOpenAI returns an AIMessage; .content holds the text
            return response.content
        except Exception as e:
            print("Error occurred:", e)
            return None

# Define specialized agent classes
class Cardiologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report=medical_report, role="Cardiologist")

class Psychologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report=medical_report, role="Psychologist")

class Pulmonologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report=medical_report, role="Pulmonologist")

class MultidisciplinaryTeam(Agent):
    def __init__(self, cardiologist_report, psychologist_report, pulmonologist_report):
        extra_info = {
            "cardiologist_report": cardiologist_report,
            "psychologist_report": psychologist_report,
            "pulmonologist_report": pulmonologist_report
        }
        super().__init__(role="MultidisciplinaryTeam", extra_info=extra_info)
