import pandas as pd
from tqdm import tqdm
import json
from argparse import ArgumentParser
from .physician_agent import PhysicianModel
from .assistant_agent import AssistantModel
from .utils import read_prompt_from_file
import chainlit as cl
import langroid.language_models as lm
import langroid as lr
from langroid.agent.callbacks.chainlit import add_instructions
import logging
from langroid.agent.callbacks.chainlit import (
    add_instructions,
    make_llm_settings_widgets,
    setup_llm,
    update_llm,
)
from textwrap import dedent
from langroid.utils.configuration import settings
from langroid.language_models import Role, LLMMessage
from rich.prompt import Prompt
from .utils import serialize_dict
from pathlib import Path


parser = ArgumentParser()
parser.add_argument('--input_file')
parser.add_argument('--history_file')
parser.add_argument('--discharge_data_file')
parser.add_argument('--mode') # check which use case is being conducted
parser.add_argument('--baseline_model')
parser.add_argument('--baseline_system_prompt')
parser.add_argument('--baseline_prompt_template')
parser.add_argument('--assistant_model')
parser.add_argument('--physician_model')
parser.add_argument('--ass_history_file')
parser.add_argument('--phy_history_file')
parser.add_argument('--phy_system_prompt')
parser.add_argument('--phy_prompt_template')
parser.add_argument('--ass_system_prompt')
parser.add_argument('--ass_prompt_template')

args = parser.parse_args()

input_file = args.input_file
history_file = args.history_file
discharge_data_file = args.discharge_data_file
use_case = args.mode
baseline_model_name = args.baseline_model
baseline_system_prompt = args.baseline_system_prompt
baseline_prompt_template = args.baseline_prompt_template
ass_model_name = args.assistant_model
phy_model_name = args.physician_model
ass_history_file = args.ass_history_file
phy_history_file = args.phy_history_file
phy_system_prompt = args.phy_system_prompt
phy_prompt_template = args.phy_prompt_template
ass_system_prompt = args.ass_system_prompt
ass_prompt_template = args.ass_prompt_template

# set info logger
logging.basicConfig(level=logging.INFO)
SRC_DIR = Path(__file__).resolve().parent
main_dir = SRC_DIR.parent
FILE = "chainlit-chat-transcript.txt"
ASS_MODEL = ass_model_name
PHY_MODEL = phy_model_name
random_seed = 42
final_discharge_text = ""
ass_message_history = ""
ass_prompt = ""
ass_model = ""
model_responses = []  
    
discharge_data = pd.read_csv(input_file)[:1]
#discharge_data = discharge_data.drop([discharge_data.index[128]])
discharge_data = discharge_data.reset_index(drop=True)
discharge_data = discharge_data[["note_id", "subject_id", "_id", "icd10_proc", "icd10_diag", "chief_complaint", "history", "physical_exam", "results", "discharge diagnosis", "discharge condition", "discharge instructions"]]
row = discharge_data.iloc[0]
        
@cl.on_chat_start
async def chat_with_chainlit():
    global ASS_MODEL
    global PHY_MODEL
    global random_seed
    global ass_prompt
    global phy_prompt  
    
    #for index, row in tqdm(discharge_data.iterrows(), total=len(discharge_data)):
    chief_complaint = str(row['chief_complaint'])
    patient_history = str(row['history'])
    physical_exam = str(row['physical_exam'])
    exam_results = str(row['results'])
            
    phy_clinical_note = "Chief complaint: " + chief_complaint
    ass_clinical_note = "Chief complaint: " + chief_complaint + "\nHistory of present illness: " + patient_history + "\nPhysical exam: " + physical_exam + "\nPertinent results: " + exam_results
            
    read_phy_system_prompt = read_prompt_from_file(phy_system_prompt)
    read_phy_prompt_template = read_prompt_from_file(phy_prompt_template)
    read_ass_system_prompt = read_prompt_from_file(ass_system_prompt)
    read_ass_prompt_template = read_prompt_from_file(ass_prompt_template)
            
    ass_model = AssistantModel(ass_model_name, system_prompt=read_ass_system_prompt, user_prompt_template=read_ass_prompt_template)
    phy_model = PhysicianModel(phy_model_name, system_prompt=read_phy_system_prompt, user_prompt_template=read_phy_prompt_template)

    ass_prompt = ass_model.generate_prompt(ass_clinical_note)
    phy_prompt = phy_model.generate_prompt(phy_clinical_note)
                    
    system_inst = """
        \n\nWhen you are ready, please directly start writing your discharge text. You should start by typing diagnosis=
        """
        
    ass_lm_config = lm.OpenAIGPTConfig(
                timeout=180,
                chat_context_length=1040_000,
                chat_model="ollama/"+ASS_MODEL,
                seed=random_seed
            )
        
    ass_agent = lr.ChatAgent(lr.ChatAgentConfig(system_message=ass_prompt,llm=ass_lm_config)) 
    cl.user_session.set("ass_agent", ass_agent)
    #lr.ChainlitAgentCallbacks(ass_agent) 
        
    await add_instructions(
            title="Welcome to Clinical Chatbot!",
            content=dedent(phy_prompt + system_inst)) 
          
    
@cl.on_message
async def on_message(message: cl.Message):
    agent: lr.ChatAgent = cl.user_session.get("ass_agent")
    lr.ChainlitAgentCallbacks(agent)
    
    global model_responses
    
    if message.content.startswith("diagnosis="):
        content = message.content
        model_responses.append(content)
        discharge_data['model_output'] = model_responses
        discharge_data.to_csv(discharge_data_file + 'output_interactive_ass_{}.csv'.format(ass_model_name),index = False)
            
        
        # get transcript of entire conv history as a string
        history = (
            "\n\n".join(
                    [
                        f"{msg.role.value.upper()}: {msg.content}"
                        for msg in agent.message_history
                    ]
                )
                + "\n\n"
                + "FINAL User Answer: "
                + content[2:]
            )

        # save chat transcript to file
        with open(SRC_DIR+FILE, "w") as f:
            f.write(f"Chat transcript:\n\n{history}\n")
            await cl.Message(
                    content=f"Chat transcript saved to {SRC_DIR+FILE}.",
                    author="System",
            ).send()
        print("[magenta]Thanks! Bye!")
        return
 
    await agent.llm_response_async(message.content)