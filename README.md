# DPQA-A-Depression-Knowledge-Graph-Question-Answering-System

🧩 Abstract

DPQA is a hybrid knowledge-graph–grounded question answering system designed for depression-related medical knowledge.
The system integrates a structured RDF knowledge graph—constructed from DrugBank, SNOMED CT, and curated Chinese medical text—with large language models (LLMs) such as GPT-4o.

DPQA automatically:

parses natural-language questions

generates SPARQL queries

executes them against GraphDB

synthesizes fact-grounded answers

Experimental results show:

Accuracy: ~75.4% on template-based questions

Hallucination rate: 0% (for supported templates)

Latency: ~2.5 seconds per query

The system reliably handles pairwise drug–drug interaction reasoning, demonstrating its potential as a trustworthy QA framework in the mental-health domain.

📂 Project Structure & Access

All project files are located under:

dpqa/
  kgqaf/
    DPQA-fallback.py                     # Main hybrid QA system (KG + LLM)
    batch_eval.py                        # Batch evaluation runner
    kgqa_templates.py                    # SPARQL template library
    kgqa_testset_full_specified.v2.json  # Evaluation set
    kgqa_eval_policy.json                # Scoring configuration
    eval_detail.json                     # Example evaluation output
    README.md
data/
  dpkg.ttl                               # RDF knowledge graph


Access to the dataset and code can be provided for academic, non-commercial research upon request.

⚙️ Installation
Step 1. Environment Setup

Requirements

Python 3.11+

GraphDB 10.6.2+

Internet connection (for OpenAI API)

Python libraries

pip install rdflib lxml pandas requests openai

Step 2. Configure Your OpenAI API Key

The hybrid system uses an LLM for:

semantic parsing

answer synthesis

Insert your API key directly into the configuration of the QA system.

Step 3. Import the Knowledge Graph

Open GraphDB → create a new repository (e.g., dpkgraph) → import:

data/dpkg.ttl


After importing, obtain the repository endpoint, such as:

http://localhost:7200/repositories/dpkgraph

Step 4. Configure the Remote GraphDB Connection

Edit DPQA-fallback.py to set:

GRAPHDB_URL = "http://<your-host>:7200/repositories/dpkgraph"


Ensure GraphDB is running during execution.

Step 5. Run the System
Interactive Mode
cd dpqa/kgqaf
python DPQA-fallback.py


The console will prompt for natural-language input (Chinese or English).
The system will:

parse

generate SPARQL

retrieve RDF results

return grounded answers

Batch Evaluation Mode
python batch_eval.py \
  --graphdb http://<host>:7200/repositories/dpkgraph \
  --templates kgqa_templates.py \
  --testset  kgqa_testset_full_specified.v2.json \
  --policy   kgqa_eval_policy.json \
  --model_parse gpt-4o \
  --model_answer gpt-4o \
  --out eval_detail.json

🧪 Evaluation & Expected Results

Using the provided dataset and settings, you should reproduce:

Template-based accuracy: ~75.4%

Average latency: ~2.5 seconds/query

Interaction reasoning: Supports pairwise drug–drug interactions via
db:interaction_description.

The evaluation output (including parsed structures, SPARQL queries, query results, and final synthesized answers) will be saved to:

eval_detail.json


Template-based results are deterministic and fully reproducible under the same configuration and prompts.
