import re
import numpy as np
import asyncio
import pandas as pd
from argparse import ArgumentParser
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics import faithfulness
from ragas.metrics.collections import FactualCorrectness
from ragas.metrics.collections import Faithfulness
from ragas.metrics.collections import AnswerRelevancy
import json
from dotenv import load_dotenv

load_dotenv()


def clean_message(msg):
    """
    Cleans unwanted patterns from message text.
    Removes internal IDs like: 16522406-DS-18):
    """
    # Remove patterns like "16522406-DS-18):"
    msg = re.sub(r"\b\d{6,}-[A-Z]{2}-\d+\):\s*", "", msg)

    # Remove leading/trailing whitespace
    msg = msg.strip()
    return msg

def parse_log(filepath:str):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    results = []

    # Regex to capture message blocks
    stop_pattern = r"(?=\n\[\d{4}-\d{2}-\d{2}|^\d{4}-\d{2}-\d{2}\s)"

    pattern = re.compile(
        r"\[(?P<timestamp>[\d\-T:\.]+)\]\s+"
        r"(?P<sender>USER|LLM)\s+MESSAGE\s+(?P<qid>Q\d+).*?:\s*"
        r"(?P<msg>.*?)(?=" + stop_pattern + r"|\Z)",
        re.DOTALL | re.MULTILINE
    )

    for match in pattern.finditer(text):
        sender = match.group("sender").lower()
        message = match.group("msg").strip()
        message = clean_message(message)
        results.append({
            "sender": sender,
            "message": message
        })

    return results


async def main():
    parser = ArgumentParser()
    parser.add_argument('--clinic_notes', help='Path to the clinic notes')
    parser.add_argument('--log_file')
    parser.add_argument('--output_file_prefix')
    parser.add_argument('--openai_model')

    args = parser.parse_args()

    # Setup LLM
    client = AsyncOpenAI()

    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client, interface="modern")
    llm = llm_factory("gpt-4o-mini", client=client, max_tokens=5000)
    factual_correctness = FactualCorrectness(llm=llm, mode="f1", atomicity="low")

    dialogue = []
    with open(args.log_file, "r", encoding="utf-8") as f:
        for line in f:
            dialogue.append(json.loads(line))

    dialogue = pd.DataFrame(dialogue)
    clinic_notes = pd.read_csv(args.clinic_notes)

    relevancy_scores = []
    faithfulness_scores = []
    factuality_scores = []
    relevancy_scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)
    faithfulness_scorer = Faithfulness(llm=llm)

    for case, row in dialogue.groupby(['case_index']):

        # TODO: Please check this one!!!
        clinic_note = clinic_notes.iloc[case].to_dict()
        chief_complaint = clinic_note['chief_complaint']
        history = clinic_note['history']
        physical_exam = clinic_note['physical_exam']
        examination_results = clinic_note['results']
        discharge_diagnosis = clinic_note['discharge diagnosis']

        examination_context = f'{chief_complaint}\n{history}\n{physical_exam}\n{examination_results}\n{discharge_diagnosis}'

        row.sort_values(by='timestamp', inplace=True)

        messages = row['message'].tolist()
        if 'final answer' in messages[-2]:
            final_answer = messages[-2]
            print(final_answer)
            print(discharge_diagnosis)
            factual_result = await factual_correctness.ascore(
                response=final_answer,
                reference = discharge_diagnosis
            )
            factuality_scores.append(factual_result.value)
            print(factuality_scores)

        for idx in range(0, len(messages),2):
            if idx+1 == len(messages):
                continue

            user_input = messages[idx]
            response = messages[idx+1]

            result = await relevancy_scorer.ascore(
                user_input=user_input,
                response = response
            )

            relevancy_scores.append({
                'user': user_input,
                'llm': response,
                'score': result.value,
            })


            result = await  faithfulness_scorer.ascore(
                user_input = user_input,
                response=response,
                retrieved_contexts=[
                    examination_context
                ]
            )

            faithfulness_scores.append({
                'user': user_input,
                'llm': response,
                'score': result.value,
            })
    factuality_scores = pd.DataFrame(factuality_scores)

    print('Average factuality scores')
    print(np.mean(factuality_scores['score'].to_numpy()))

    factuality_scores.to_csv(f'{args.output_file_prefix}_factuality_scores.csv')

    relevancy_scores = pd.DataFrame(relevancy_scores)

    print('Average relevancy scores')
    print(np.mean(relevancy_scores['score'].to_numpy()))

    relevancy_scores.to_csv(f'{args.output_file_prefix}_relevancy_scores.csv')

    faithfulness_scores = pd.DataFrame(faithfulness_scores)

    print('Average faithfulness scores')
    print(np.mean(faithfulness_scores['score'].to_numpy()))

    faithfulness_scores.to_csv(f'{args.output_file_prefix}_faithfulness_scores.csv')
#
asyncio.run(main())


