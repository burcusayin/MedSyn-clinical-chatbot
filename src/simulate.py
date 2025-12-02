import pandas as pd
from tqdm import tqdm
import json
from argparse import ArgumentParser
from src.physician_agent import PhysicianModel
from src.assistant_agent import AssistantModel
from src.baseline_agent import BaselineModel
from src.langroid_endpoint import MainChatAgent
from src.utils import read_prompt_from_file
        
def main():
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
    parser.add_argument(
        '--num_rows',
        type=int,
        default=None,
        help='If set, only use the first N rows from the input file (for quick tests).',
    )

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
    num_rows = args.num_rows

    # Load data
    discharge_data = pd.read_csv(input_file)
    if num_rows is not None:
        discharge_data = discharge_data.head(num_rows)

    discharge_data = discharge_data.reset_index(drop=True)
    discharge_data = discharge_data[
        [
            # adapt these to your actual CSV columns:
            "Dataset",
            "note_id",
            "Difficulty",
            "chief_complaint",
            "history",
            "physical_exam",
            "results",
            "discharge diagnosis",
        ]
    ]

    model_responses = []
    if use_case == "phy_baseline":
        print("Use case is phy_baseline...")
        hist = {}
        for index, row in tqdm(discharge_data.iterrows(), total=len(discharge_data)):
            chief_complaint = str(row['chief_complaint'])
            clinical_note = "Chief complaint: " + chief_complaint
            baseline_system = read_prompt_from_file(baseline_system_prompt)
            baseline_prompt = read_prompt_from_file(baseline_prompt_template)
            model = BaselineModel(baseline_model_name, system_prompt=baseline_system, user_prompt_template=baseline_prompt)
            prompt = model.generate_prompt(clinical_note)
            chatAgent = MainChatAgent(chat_mode="baseline", phy_model_id=baseline_model_name, phy_prompt=prompt)
            print("Index:", index)
            response, history = chatAgent.start_chat()
            model_responses.append(response)
            hist[index] = str(history)
        discharge_data["model_output"] = model_responses
        discharge_data.to_csv(discharge_data_file + 'output_phy_baseline_{}.csv'.format(baseline_model_name.split('/')[-1]),index = False)
        histJson = history_file + "{}_{}.json".format(use_case, baseline_model_name)
        with open(histJson, "w") as outfile: 
            json.dump(hist, outfile)
    elif use_case == "ass_baseline":
        print("Use case is ass_baseline...")
        hist = {}  
        for index, row in tqdm(discharge_data.iterrows(), total=len(discharge_data)):
            chief_complaint = str(row['chief_complaint'])
            patient_history = str(row['history'])
            physical_exam = str(row['physical_exam'])
            exam_results = str(row['results'])
            clinical_note = "Chief complaint: " + chief_complaint + "\nHistory of present illness: " + patient_history + "\nPhysical exam: " + physical_exam + "\nPertinent results: " + exam_results
            baseline_system = read_prompt_from_file(baseline_system_prompt)
            baseline_prompt = read_prompt_from_file(baseline_prompt_template)
            model = BaselineModel(baseline_model_name, system_prompt=baseline_system, user_prompt_template=baseline_prompt)
            prompt = model.generate_prompt(clinical_note)
            chatAgent = MainChatAgent(chat_mode="baseline", phy_model_id=baseline_model_name, phy_prompt=prompt)
            print("Index:", index)
            response, history = chatAgent.start_chat()
            model_responses.append(response)
            hist[index] = str(history)
        discharge_data["model_output"] = model_responses
        discharge_data.to_csv(discharge_data_file + 'output_ass_baseline_{}.csv'.format(baseline_model_name.split('/')[-1]),index = False)
        histJson = history_file + "{}_{}.json".format(use_case, baseline_model_name.split('/')[-1])
        with open(histJson, "w") as outfile: 
            json.dump(hist, outfile)
    elif use_case == "interactive":
        print("Use case is interactive dialogue...")
        ass_dial_hist = {}
        
        for index, row in tqdm(discharge_data.iterrows(), total=len(discharge_data)):
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

            ass_prompt_generated = ass_model.generate_prompt(ass_clinical_note)
            phy_prompt_generated = phy_model.generate_prompt(phy_clinical_note)
            
            print("Index:", index)
            chatAgent = MainChatAgent(chat_mode="interactive", phy_model_id=phy_model_name, ass_model_id=ass_model_name, phy_prompt=phy_prompt_generated, ass_prompt=ass_prompt_generated)
            response, ass_history = chatAgent.start_interactive_chat()

            model_responses.append(response)
            ass_dial_hist[index] = str(ass_history)
            
        discharge_data['model_output'] = model_responses
        discharge_data.to_csv(discharge_data_file + 'output_interactive_ass_{}.csv'.format(ass_model_name.split('/')[-1]),index = False)
        
        assHistJson = ass_history_file + "{}_ass_{}.json".format(use_case, ass_model_name.split('/')[-1])
        with open(assHistJson, "w") as outfile: 
            json.dump(ass_dial_hist, outfile)
    
    else:
        print("Use case is two-agent dialogue...")
        phy_dial_hist = {}
        ass_dial_hist = {}
        
        for index, row in tqdm(discharge_data.iterrows(), total=len(discharge_data)):
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

            ass_prompt_generated = ass_model.generate_prompt(ass_clinical_note)
            phy_prompt_generated = phy_model.generate_prompt(phy_clinical_note)
            
            print("Index:", index)
            chatAgent = MainChatAgent(chat_mode="two-agent", phy_model_id=phy_model_name, ass_model_id=ass_model_name, phy_prompt=phy_prompt_generated, ass_prompt=ass_prompt_generated)
            response, ass_history, phy_history = chatAgent.start_chat()

            model_responses.append(response)
            
            phy_dial_hist[index] = str(phy_history)
            ass_dial_hist[index] = str(ass_history)
            
        discharge_data['model_output'] = model_responses
        discharge_data.to_csv(discharge_data_file + 'output_phy_{}_ass_{}.csv'.format(phy_model_name.split('/')[-1],ass_model_name.split('/')[-1]),index = False)
        
        phyHistJson = phy_history_file + "{}_phy_{}_ass_{}.json".format(use_case, phy_model_name.split('/')[-1], ass_model_name.split('/')[-1])
        with open(phyHistJson, "w") as outfile: 
            json.dump(phy_dial_hist, outfile)
        
        assHistJson = ass_history_file + "{}_ass_{}_phy_{}.json".format(use_case, ass_model_name.split('/')[-1], phy_model_name.split('/')[-1])
        with open(assHistJson, "w") as outfile: 
            json.dump(ass_dial_hist, outfile)
            
if __name__ == "__main__":
    main()