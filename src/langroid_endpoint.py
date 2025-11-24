import langroid as lr
import langroid.language_models as lm
import logging
from pydantic import BaseModel, Field 
from langroid.utils.configuration import settings
from langroid.agent.tools.orchestration import ForwardTool, ResultTool
from langroid.agent.task import TaskConfig
from .utils import serialize_dict
from langroid.language_models import Role, LLMMessage
from rich.prompt import Prompt 


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
lr.utils.logging.setup_colored_logging()

CP_NAME = "CP"
PA_NAME = "PA"


class DischargeText(BaseModel):
    diagnosis: str = Field(..., description="final diagnosis relevant to the patient's condition at the time of discharge")
    codes: str = Field(..., description="ICD-10 codes corresponding to the patient's diagnosis")     
    
class BaselineDischargeTextTool(lr.ToolMessage):
    """Write the discharge text for a patient"""

    request: str = "baseline_discharge_text_tool"
    purpose: str = """
    Write the <dischargeText>,
    with all fields of the appropriate type filled out;
    SIMPLY TALK IN NATURAL LANGUAGE.
    """
    dischargeText: DischargeText
    
    def handle(self) -> ResultTool:
        """Handle LLM's structured output if it matches DischargeText structure"""
        print("SUCCESS! Got Valid DischargeText Info")
        
        return ResultTool(
            status = "**DONE!**",
            dischargeText = self.dischargeText
        )
        
    @staticmethod
    def handle_message_fallback(agent: lr.ChatAgent, msg: str | lr.ChatDocument) -> str:
        """
        We end up here when there was no recognized tool msg from the LLM;
        In this case we remind the agent to use discharge_text_tool.
        """ 

        return f"""You must use the TOOL {BaselineDischargeTextTool.name()} to write the dischargeText. Please check the required format carefully, and do not forget to include "request": "baseline_discharge_text_tool".  Remember that the dischargeText should include `diagnosis` and `codes` fields  only. Both fields must be strings; do not add [] for the codes."""

        #llama models
        #return f"""You must use the TOOL {BaselineDischargeTextTool.name()} to write the dischargeText. Please check the required format carefully, and do not forget to close the brackets you used."""
    
# Define the Tool class for the LLM to use, to produce the above structure.
class DischargeTextTool(lr.ToolMessage):
    """Write the discharge text for a patient"""

    request: str = "discharge_text_tool"
    purpose: str = """
    To write the final <dischargeText> AFTER having a multi-turn discussion about the patient with your assistant,
    with all fields of the appropriate type filled out;
    SIMPLY TALK IN NATURAL LANGUAGE.
    """
    dischargeText: DischargeText
    
    def handle(self) -> ResultTool:
        """Handle LLM's structured output if it matches DischargeText structure"""
        print("SUCCESS! Got Valid DischargeText Info")
        
        return ResultTool(
            status = "**DONE!**",
            dischargeText = self.dischargeText
        )

    @staticmethod
    def handle_message_fallback(
        agent: lr.ChatAgent, msg: str | lr.ChatDocument
    ) -> ForwardTool:
        """
        We end up here when there was no recognized tool msg from the LLM;
        In this case forward the message to the PA agent using ForwardTool.
        """
        if isinstance(msg, lr.ChatDocument) and msg.metadata.sender == lr.Entity.LLM:
            return ForwardTool(agent=PA_NAME)
        # else:
        #     return f"""You must use the TOOL {DischargeTextTool.name()} to write the dischargeText. Please check the required format carefully, and do not forget to close the brackets you used."""


class MainChatAgent:
    def __init__(self, chat_mode:str, phy_model_id:str, ass_model_id:str="", phy_prompt:str="", ass_prompt:str="",
                    d: bool = False,  # pass -d to enable debug mode (see prompts etc)
                    nc: bool = False,  # pass -nc to disable cache-retrieval (i.e. get fresh answers)
                    random_seed: float=42
                ):
        self.chat_mode = chat_mode
        self.phy_model_id = phy_model_id
        self.ass_model_id = ass_model_id
        self.random_seed = random_seed
        settings.debug = d
        settings.cache = not nc
        
        if self.chat_mode == "baseline":
            self.create_single_agent(phy_prompt)
        elif self.chat_mode == "interactive":
            self.create_interactive_agent(phy_prompt, ass_prompt)
        else:
            self.create_two_agents(phy_prompt, ass_prompt)
        
    def create_single_agent(self, phy_prompt:str):
        print("Creating single agent environment...")
        self.phy_lm_config = lm.OpenAIGPTConfig(chat_model="ollama/"+self.phy_model_id,chat_context_length=1040_000, seed=self.random_seed)
        self.phy_agent = lr.ChatAgent(lr.ChatAgentConfig(name=CP_NAME, llm=self.phy_lm_config, system_message=phy_prompt)) 
        self.phy_agent.enable_message(BaselineDischargeTextTool)
    
    def create_interactive_agent(self, phy_prompt, ass_prompt):
        self.ass_lm_config = lm.OpenAIGPTConfig(chat_model="ollama/"+self.ass_model_id,chat_context_length=1040_000, seed=self.random_seed)
        self.ass_agent = lr.language_models.OpenAIGPT(self.ass_lm_config)
        self.ass_prompt = ass_prompt
        self.phy_prompt = phy_prompt
        
    def create_two_agents(self, phy_prompt:str, ass_prompt:str=""):
        print("Creating  two-agent environment...")
        #tried setting temperature=0 but did not work
        self.ass_lm_config = lm.OpenAIGPTConfig(chat_model="ollama/"+self.ass_model_id,chat_context_length=1040_000, seed=self.random_seed)
        self.ass_agent = lr.ChatAgent(lr.ChatAgentConfig(name=PA_NAME, llm=self.ass_lm_config, system_message=ass_prompt)) 
        self.phy_lm_config = lm.OpenAIGPTConfig(chat_model="ollama/"+self.phy_model_id,chat_context_length=1040_000, seed=self.random_seed)
        self.phy_agent = lr.ChatAgent(lr.ChatAgentConfig(name=CP_NAME, llm=self.phy_lm_config, system_message=phy_prompt)) 
        self.phy_agent.enable_message(DischargeTextTool)
        
    def format_history(self, agent_history):
        processed_history = []
        for idx, conv in enumerate(agent_history):
            processed_history.append(serialize_dict(conv.model_dump()))
        return processed_history
        
    def start_chat(self):
        task_config = TaskConfig(inf_loop_cycle_len=0)
        if self.chat_mode == "baseline":
            print("baseline task is running")
            self.phy_task = lr.Task(self.phy_agent, 
                                    interactive=False,
                                    single_round=False,
                                    restart=True,
                                    allow_null_result=False,
                                    config=task_config)[ResultTool]  # specialize task to strictly return ResultTool
            response_tool: ResultTool = self.phy_task.run()  
            print("response_tool: ", response_tool)
            msg_history = self.format_history(self.phy_agent.message_history)
            
            if response_tool is None:
                print("""RETURNED ANSWER DOES NOT HAVE A TOOL! LLM DID NOT FORMAT THE DISCHARGE TEXT!!!""")
                return None, msg_history
            else:
                print("ResultTool has been received successfully!!!")
                print(response_tool.dischargeText)
                return response_tool.dischargeText, msg_history
                
        else:
            print("two-agent task is running")
            self.ass_task = lr.Task(self.ass_agent, 
                                    llm_delegate=True,
                                    interactive=False,
                                    single_round=True,
                                    config=task_config)
            
            self.phy_task = lr.Task(self.phy_agent, 
                                    llm_delegate=True,
                                    interactive=False,
                                    single_round=False,
                                    config=task_config)[ResultTool]  # specialize task to strictly return ResultTool
            
            self.phy_task.add_sub_task(self.ass_task)
            response_tool: ResultTool = self.phy_task.run(turns=100)  
            print("response_tool: ", response_tool)
            
            ass_message_history = self.format_history(self.ass_agent.message_history)
            phy_message_history = self.format_history(self.phy_agent.message_history)
            
            if response_tool is None:
                print("""RETURNED ANSWER DOES NOT HAVE A TOOL! LLM DID NOT FORMAT THE DISCHARGE TEXT!!!""")
                return None, ass_message_history, phy_message_history
                #return DischargeText(diagnosis="unknown", condition="unknown", instructions="null"), ass_message_history, phy_message_history 
            else:
                print("ResultTool has been received successfully!!!")
                print(response_tool.dischargeText)
                return response_tool.dischargeText, ass_message_history, phy_message_history
        
    def start_interactive_chat(self):
        system_inst = """
        Enter R when you are ready to write the discharge text. 
        Enter x or q to quit without writing the discharge text.
        """
        phy_system_msg = self.phy_prompt + system_inst
        print(phy_system_msg)
        messages = [LLMMessage(content=self.ass_prompt, role=Role.SYSTEM), ]
        while True:
            message = Prompt.ask("[blue]Physician")
            messages.append(LLMMessage(role=Role.USER, content=message))
            
            if message in ["x", "q"]:
                print("[magenta]Bye!")
                break
            elif message in ["R"]:
                message = Prompt.ask("Please write the discharge text now!...\n [blue]Physician")
                messages.append(LLMMessage(role=Role.USER, content=message))
                print("LLMMessage(role=Role.USER, content=message): ", LLMMessage(role=Role.USER, content=message))
                dischargeText = str(LLMMessage(role=Role.USER, content=message)).split('Role.USER : ')[1]
                print("[magenta]Thanks! Bye!")
                break
            response = self.ass_agent.chat(messages=messages, max_tokens=200)
            messages.append(response.to_LLMMessage())
            print("[green]Assistant: " + response.message)
            
        ass_message_history = self.format_history(messages)
        return dischargeText, ass_message_history
