"""Ask a question to the notion database."""
import json
import os
import pickle
import pandas as pd
from langchain import hub
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from ruamel.yaml import YAML


def get_retriever():
    with open("params.yaml") as f:
        params = YAML().load(f)
    emb_params = params['Embeddings']
    emb = HuggingFaceEmbeddings(**emb_params)

    store = FAISS.load_local("data/docs.index", emb, allow_dangerous_deserialization=True)
    return store.as_retriever()


def get_llm():
    with open("params.yaml") as f:
        params = YAML().load(f)
    chat_params = params['ChatLLM']
    return HuggingFaceEndpoint(**chat_params)


def get_prompt():
    return hub.pull("rlm/rag-prompt").messages[0].prompt


def chain(question, retriever, llm, prompt):
    context = retriever.get_relevant_documents(question)
    context = [doc.page_content for doc in context]
    context_str = "\n\n".join(context)
    print(f"Question: {question}")

    input = prompt.invoke({"question": question, "context": context_str})
    result = llm.invoke(input.text) 
    return context, result


if __name__ == '__main__':
    retriever = get_retriever()
    llm = get_llm()
    prompt = get_prompt()

    df = pd.read_csv("data/ground_truths.csv")
    sample_questions = df["Q"].to_list()

    records = []
    for question in sample_questions:
        context, result = chain(question, retriever, llm, prompt)
        records.append({
            "Q": question,
            "A": result,
            "context": context,
        })

        print(f"Answer: {result}")
        print("\n\n")

    with open("data/results.json", "w") as f:
        json.dump(records, f)
