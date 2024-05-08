import json
from ruamel.yaml import YAML

import pandas as pd
from datasets import Dataset
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import metrics
from ragas import evaluate


def get_eval_dataset():
    with open("data/results.json") as f:
        results = json.load(f)

    questions, answers, contexts = [], [], []
    for row in results:
        questions.append(row["Q"])
        answers.append(row["A"])
        contexts.append(row["context"])

    truth = pd.read_csv("data/ground_truths.csv")
    ground_truth = truth["A"].to_list()

    return Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truth
            })


def run_eval(dataset):
    with open("params.yaml") as f:
        params = YAML().load(f)
    emb_params = params['Embeddings']
    chat_params = params['ChatLLM']

    emb = HuggingFaceEmbeddings(**emb_params)
    llm = HuggingFaceHub(**chat_params)

    result = evaluate(
        dataset,
        metrics=[metrics.answer_similarity],
        llm=llm,
        embeddings=emb,
    )
    return result


if __name__ == '__main__':
    dataset = get_eval_dataset()
    result = run_eval(dataset)
    print(result)
    df = result.to_pandas()
    df.to_csv("data/eval.csv", header=True, index=False)
