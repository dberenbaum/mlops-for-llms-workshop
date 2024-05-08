# Using DVC to run the pipeline

[DVC](http://dvc.org) enables fast iteration on the RAG application in a few ways:
1. Encourages modularizing and parametrizing code, making it easy to make changes
   without affecting other parts of the application.
2. Runs pipelines across multiple files, keeping track of what's changed so you don't
   have to manually re-run parts of the code or think about what steps need to run
   again.
3. Tracks outputs so you can recover the results from any prior iteration.

### 1. Modularizing and parametrizing code

This is not specific to DVC but has benefits for any codebase. The notebook code has
been divided into reusable scripts in `src`, and the parameters have been extracted to
`params.yaml`. This means you can try out different parameters by simply modifying the
values in `params.yaml`, or change the code without having to worry about copying those
changes elsewhere.

### 2. Running pipelines

A DVC pipeline is defined in `dvc.yaml`, which will run each script according to the
dependencies and outputs defined in the stages there. Run the pipeline with the command
`dvc repro`. The first time the pipeline is
run, every stage will be run in order, and the hashes of the dependencies and outputs of
each stage will be saved in `dvc.lock`. Subsquent runs will only run those stages for
which dependencies have changed. Parameters are also
defined as dependencies so that only the relevant parameters will trigger stages to run.

### 3. Tracking outputs

The output of each stage is tracked by DVC, which works alongside Git. If you try out a new approach but find the
results were better before, you can run `git checkout` followed by `dvc checkout` to 
retrieve the previous outputs without having to re-run the code. This is especially
important as the pipeline gets longer and the data get larger.

## Example: Use embeddings for retrieval

As an example, let's replace the simple BM25 retriever with an embeddings-based
retriever. To use embeddings, first calculate the embeddings and save them to
a vector store. This is implemented in `src/vectore_store.py`, which uses 
https://huggingface.co/sentence-transformers/all-mpnet-base-v2 as the embedding model and https://github.com/facebookresearch/faiss as the vector store.

To use the embeddings, update the `get_retriever()` function in `src/qa.py`:

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_retriever():
    with open("params.yaml") as f:
        params = YAML().load(f)
    emb_params = params['Embeddings']
    emb = HuggingFaceEmbeddings(**emb_params)

    store = FAISS.load_local("docs.index", emb, allow_dangerous_deserialization=True)
    return store.as_retriever()
```

This is already done in `src/qa_embeddings.py`.

To update the DVC pipeline, add a stage in `dvc.yaml` to run `vectore_store.py`:

```yaml
  vectorize:
    cmd: python src/vector_store.py
    params:
    - Embeddings
    deps:
    - docs.json
    - src/vector_store.py
    outs:
    - docs.index
```

Also edit the stage that runs `src/qa.py` to add a couple of new dependencies since it now relieas on the embeddings model parameters and the saved vector store index (`docs.index`):

```diff
   run:
    cmd: python src/qa.py
     params:
+    - Embeddings
     - ChatLLM
     deps:
+    - docs.index
     - ground_truths.csv
     - src/qa.py
     outs:
     - results.json
```

This is already done in `dvc_embeddings.yaml`, so you can alternatively copy that file to `dvc.yaml`.
Run `dvc repro` to get the updated results.

## Exercise

For this exercise, take the code for the DVC RAG application you built in the notebooks and convert it into a DVC pipeline.

It is recommended that you first commit all the work you have done so far to Git. Optionally, you may want to create a new branch for this exercise with `git checkout -b dvc`. Then, you can modify the existing pipeline and Python files without worrying about messing up anything.

To complete the exercise, you will need to:
1. Convert the code you wrote in the notebooks into Python files. You can start either by copying the notebook cells into Python files or by modifying the existing Python files.
2. Update your code and `params.yaml` to include any parameters that you want to define outside the code, like chunk size, model ID, etc.
3. Update `dvc.yaml` to define the pipeline of steps to run. For each step, define the command, parameters, dependencies, and outputs.
4. Run `dvc repro` and validate that you can run the pipeline end-to-end.

Now that you have a working pipline, you can try to make changes and run it again to see how it impacts the results.