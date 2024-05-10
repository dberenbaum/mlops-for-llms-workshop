# MLOps for LLMs Workshop

This repo builds a Git question-answering chat bot. The goal is both to show how to build such a bot but also how MLOps can help build and iterate on such applications.

This chat bot is built on top of LangChain and uses the [Pro Git
book](https://git-scm.com/book/en/v2) as documentation.

This is a chatbot about Git where the training pipeline was built using DVC.

It was initially inspired by https://github.com/hwchase17/notion-qa.

# Environment Setup

First you need to do a git pull of the code:
```shell
git clone git@github.com:iterative/llm-demo.git
cd llm-demo
```

You also need [Anaconda](https://www.anaconda.com/download/success) to install the
environment (note: the FAISS dependency will not work without Anaconda).

In order to set your environment up to run the code here, first install all requirements in a conda env:
```shell
conda create -n mlops-for-llms-workshop --python=python3.11
conda activate mlops-for-llms-workshop
pip install -r requirements.txt
```

Then set your Hugging Face API key (if you don't have one, get one
[here](https://huggingface.co/docs/hub/en/security-tokens)):
```shell
  export HUGGINGFACEHUB_API_TOKEN=....
```
The preceeding spaces prevent the API key from staying in your bash history if that is [configured](https://stackoverflow.com/questions/6475524/how-do-i-prevent-commands-from-showing-up-in-bash-history).

# Running

Now you should be ready to run any code in the repo.

You can start by exploring the notebooks are in `notebooks`, or run the whole pipeline in `src` using [DVC](dvc.md):
```shell
$ dvc repro
```
The pipeline is set up to use a simple BM25 retriever, but you can replace it with an
embeddings-based retriever by replacing the `dvc.yaml` file:
```shell
$ cp dvc_embeddings.yaml dvc.yaml
```

There is also a demo web UI you can start using:
```shell
$ streamlit run src/main.py
```
The log of interactions can be found in `data/chat.log`.
