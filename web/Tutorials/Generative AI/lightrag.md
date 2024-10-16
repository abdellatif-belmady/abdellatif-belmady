---
comments: true
---

# LightRAG Documentation

## Installation

### Install from source

```bash
cd LightRAG
pip install -e .
```

## Quick Start

1. Download the demo text "A Christmas Carol by Charles Dickens":

```bash
curl https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt > ./book.txt
```

2. Create a new file `lightrag.py` in your project directory and add the following code:

```python
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete
import os

os.environ["OPENAI_API_KEY"] = ""

WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)

with open("./book.txt", encoding='utf-8') as f:
    rag.insert(f.read())

# Perform naive search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="naive")))

# Perform local search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="local")))

# Perform global search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="global")))

# Perform hybrid search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid")))
```

3. Run the script:

```bash
python lightrag.py
```

## Using Hugging Face Models

To use Hugging Face models with LightRAG, configure it as follows:

```python
from lightrag.llm import hf_model_complete, hf_embedding
from transformers import AutoModel, AutoTokenizer

# Initialize LightRAG with Hugging Face model
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,  # Use Hugging Face complete model for text generation
    llm_model_name='meta-llama/Llama-3.1-8B-Instruct',  # Model name from Hugging Face
    # Use Hugging Face embedding function
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=5000,
        func=lambda texts: hf_embedding(
            texts, 
            tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
            embed_model=AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        )
    ),
)
```

## Advanced Usage

### Batch Insert

Insert multiple texts at once:

```python
rag.insert(["TEXT1", "TEXT2", ...])
```

### Incremental Insert

Insert new documents into an existing LightRAG instance:

```python
rag = LightRAG(working_dir="./dickens")

with open("./newText.txt") as f:
    rag.insert(f.read())
```

## Evaluation

### Dataset

The dataset used in LightRAG can be downloaded from [TommyChien/UltraDomain](https://huggingface.co/datasets/TommyChien/UltraDomain).

### Generate Query

LightRAG uses the following prompt to generate high-level queries. The corresponding code is located in `example/generate_query.py`.

```
Given the following description of a dataset:

{description}

Please identify 5 potential users who would engage with this dataset. For each user, list 5 tasks they would perform with this dataset. Then, for each (user, task) combination, generate 5 questions that require a high-level understanding of the entire dataset.

Output the results in the following structure:
- User 1: [user description]
    - Task 1: [task description]
        - Question 1:
        - Question 2:
        - Question 3:
        - Question 4:
        - Question 5:
    - Task 2: [task description]
        ...
    - Task 5: [task description]
- User 2: [user description]
    ...
- User 5: [user description]
    ...
```

### Batch Eval

To evaluate the performance of two RAG systems on high-level queries, LightRAG uses the following prompt. The specific code is available in `example/batch_eval.py`.

```
---Role---
You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

---Goal---
You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**. 

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

Here is the question:
{query}

Here are the two answers:

**Answer 1:**
{answer1}

**Answer 2:**
{answer2}

Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format:

{
    "Comprehensiveness": {
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    },
    "Diversity": {
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    },
    "Empowerment": {
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    },
    "Overall Winner": {
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
    }
}
```
