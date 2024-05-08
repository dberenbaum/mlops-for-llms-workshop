import faiss
import pickle
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from ruamel.yaml import YAML


def gen_vector_store(docs):
    with open("params.yaml") as f:
        params = YAML().load(f)
    emb_params = params['Embeddings']

    emb = HuggingFaceEmbeddings(**emb_params)

    return FAISS.from_texts(docs, emb)


if __name__ == '__main__':
    with open("data/docs.json", "r") as f:
        docs = json.load(f)

    print(f"Processing {len(docs)} documents.")

    store = gen_vector_store(docs)
    store.save_local("data/docs.index")
