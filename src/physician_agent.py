class PhysicianModel:
    def __init__(self, model_id:str, system_prompt: str, user_prompt_template: str):
        self.model_id = model_id
        self.system_message = system_prompt
        self.user_prompt_template = user_prompt_template
    
    def generate_prompt(self, clinicalNote: str):
        prompt = self.user_prompt_template.format(clinicalNote=clinicalNote)
        return self.system_message + prompt
    
