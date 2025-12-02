class PhysicianModel:
    def __init__(self, model_id:str, system_prompt: str, user_prompt_template: str):
        self.model_id = model_id
        self.system_message = system_prompt
        self.user_prompt_template = user_prompt_template
    
    def generate_prompt(self, clinicalNote: str | None = None):
        """
        Generate the physician prompt.

        - If `clinicalNote` is provided, we try to inject it into the
          user prompt template using the {clinicalNote} placeholder.
        - If it is not provided, we fall back to the bare template.
        """
        if clinicalNote is not None:
            # If the template doesn't contain {clinicalNote}, .format()
            # will just leave the string unchanged.
            prompt = self.user_prompt_template.format(clinicalNote=clinicalNote)
        else:
            prompt = self.user_prompt_template

        return self.system_message + prompt
    
