# SkillOntology

## Ontology-Guided Skill Retrieval for LLM Agents

Code and data accompanying **SkillOntology: Ontology-Guided Skill Retrieval for LLM Agents**.

[Paper](SkillOntology.pdf)

This repository provides the shared core implementation, benchmark skillsets, fixed evaluation task manifests, and prebuilt ontology and MetaOp graph assets for ALFWorld, WebShop, and ScienceWorld.

## Overview

SkillOntology connects reusable procedural knowledge with the requirements of individual tasks. It represents procedures as **meta-operations (MetaOps)** and organizes them using domain concepts and supported relations.

The framework has three main components:

- **Domain Ontology:** shared concepts, state descriptions, and rules induced from public training interactions.
- **Global MetaOp Graph:** reusable operations extracted from skill documents and reconciled through extraction, verification, and adjudication.
- **Task-conditioned MetaOp Graph:** selected operation instances and dependencies rendered as guidance for an acting LLM.

## Framework

```text
Public training interactions
            |
            v
     Domain Ontology ---> Extraction Policy
                                  |
Skill documents ------------------+
                                  v
                  Extractor / Verifier / Adjudicator
                                  |
                                  v
                        Global MetaOp Graph
                                  |
Task + initial public observation |
            |                     |
            +----------+----------+
                       v
              Retrieval and compilation
                       |
                       v
           Task-conditioned MetaOp Graph
                       |
                       v
               Procedural guidance
                       |
                       v
          LLM actions <--> Benchmark environment
```

## Repository Structure

```text
SkillOntology/
├── skillontology/
│   ├── construction/
│   ├── ontology/
│   ├── retrieval/
│   ├── graph/
│   ├── rendering/
│   ├── runtime/
│   ├── benchmark/
│   │   ├── alfworld/
│   │   ├── webshop/
│   │   └── scienceworld/
│   ├── schema.py
│   └── cli.py
├── configs/
├── datasets/
│   ├── skillsets/
│   └── testsets/
├── assets/
│   ├── alfworld/
│   ├── webshop/
│   └── scienceworld/
├── pyproject.toml
├── requirements.txt
├── SkillOntology.pdf
└── README.md
```

The three benchmark directories use the same small structure: `adapter.py` handles task representations and `environment.py` connects to the benchmark. Construction, retrieval, graph compilation, rendering, and the interaction loop are organized in shared modules.

## Data and Prebuilt Assets

### Skill Source

The source skill documents for all three benchmarks come from **Graph of Skills (GoS)** and are distributed through the Hugging Face dataset [davidliuk/graph-of-skills-data](https://huggingface.co/datasets/davidliuk/graph-of-skills-data). We use benchmark-specific subsets of these publicly released skill libraries: **37 ALFWorld skills, 23 WebShop skills, and 59 ScienceWorld skills**.

The original skill documents are provided by GoS. SkillOntology constructs domain ontologies from public training interactions and derives MetaOps, operation graphs, and their vector indexes from the selected skills. The evaluation task manifests select tasks from the corresponding benchmark environments.

| Benchmark | Skill documents | Evaluation tasks | MetaOps | Global edges |
| --- | ---: | ---: | ---: | ---: |
| ALFWorld | 37 | 134 | 35 | 55 |
| WebShop | 23 | 200 | 38 | 19 |
| ScienceWorld | 59 | 100 | 106 | 111 |

- `datasets/skillsets/<benchmark>/` contains the benchmark-specific skill subsets sourced from the GoS dataset linked above.
- `datasets/testsets/<benchmark>.json` contains public task instructions and the native task identities used to select evaluation episodes.
- `assets/<benchmark>/` contains the prebuilt domain ontology, extraction policy, MetaOps, global edges, source mappings, and vector index.

The evaluation manifests identify tasks in the original benchmark environments. Install the corresponding environment and its data separately using the upstream setup instructions.

## Installation

Python **3.11** is recommended. The package declares support for Python 3.10–3.12. The commands below use Bash and should be run from the repository root.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Install the optional environment dependencies for the benchmarks you want to run:

```bash
python -m pip install -e ".[alfworld,scienceworld]"
```

### Benchmark Environments

| Benchmark | Setup | Additional configuration |
| --- | --- | --- |
| ALFWorld | Follow the [official installation and data instructions](https://github.com/alfworld/alfworld). | Set `ALFWORLD_DATA_DIR` to the data directory and `ALFWORLD_BASE_CONFIG` to the benchmark YAML configuration. |
| WebShop | Follow the [official repository setup](https://github.com/princeton-nlp/WebShop), including the full product catalog, instruction data, search index, and environment dependencies. | Set `WEBSHOP_REPO` to the checkout directory and start Python with `PYTHONHASHSEED=0`. |
| ScienceWorld | Use SDK **1.2.3**, installed by the optional dependency above, and prepare Java following the [official instructions](https://github.com/allenai/ScienceWorld). | The included configuration uses the `easy` simplification. |

Set the paths needed by your selected benchmark, replacing the placeholders with your local environment locations:

```bash
export ALFWORLD_DATA_DIR="<alfworld-data-directory>"
export ALFWORLD_BASE_CONFIG="<alfworld-base-config.yaml>"
export WEBSHOP_REPO="<webshop-checkout-directory>"
```

### Model Configuration

The supplied configurations use `deepseek-v4-flash` for chat and `text-embedding-3-large` with 3,072 dimensions for embeddings.

Set the API credentials in your shell:

```bash
export SKILLONTOLOGY_CHAT_API_KEY="<chat-api-key>"
export OPENAI_API_KEY="<embedding-api-key>"
export BASE_URL="https://api.deepseek.com"
```

`BASE_URL` supplies the WebShop chat endpoint. The ALFWorld and ScienceWorld configurations already contain this endpoint. A common endpoint override can be supplied through `SKILLONTOLOGY_CHAT_BASE_URL`.

Model names and generation parameters are configured in `configs/<benchmark>.json`. Query embeddings must use the same model and dimension as the prebuilt vector index.

## Quick Start

### Check the Included Data and Index

```bash
python -m skillontology check --config configs/alfworld.json
python -m skillontology check --config configs/webshop.json
python -m skillontology check --config configs/scienceworld.json
```

Each command reports the task count, MetaOp count, and embedding dimension. These checks load the local assets without requesting model completions or running environment episodes.

### Run an Evaluation Task

After preparing the environment and API credentials, run a task from the corresponding manifest:

```bash
python -m skillontology run --config configs/alfworld.json --task-index 0
PYTHONHASHSEED=0 python -m skillontology run --config configs/webshop.json --task-index 0
python -m skillontology run --config configs/scienceworld.json --task-index 0
```

`--task-index` is a zero-based position in the manifest's `records` array. Each invocation runs one episode and prints its task identity, success, score, stop reason, action-proposal count, environment-call count, and recorded chat-token usage as JSON. Running episodes makes chat and query-embedding API requests.

### Build a MetaOp Workspace

Build a new workspace from the included skill documents and the corresponding frozen domain ontology:

```bash
python -m skillontology build --config configs/alfworld.json --output build/alfworld --agent-mode full
python -m skillontology build --config configs/webshop.json --output build/webshop --agent-mode full
python -m skillontology build --config configs/scienceworld.json --output build/scienceworld --agent-mode full
```

The `full` mode enables extraction, verification, and adjudication and makes model API requests. Outputs are written to the directory specified by `--output`. The run commands above use the supplied prebuilt assets.

## Command-Line Arguments

| Argument | Command | Description |
| --- | --- | --- |
| `--config` | `check`, `run`, `build` | Benchmark configuration file. |
| `--task-index` | `run` | Zero-based evaluation task position. |
| `--output` | `build` | Directory for the new MetaOp workspace. |
| `--agent-mode` | `build` | `deterministic`, `extract_only`, or `full`; defaults to `full`. |

## Runtime Configuration

| Setting | ALFWorld | WebShop | ScienceWorld |
| --- | ---: | ---: | ---: |
| Temperature | 0 | 0 | 0 |
| Thinking | Disabled | Disabled | Disabled |
| Interaction history | 6 rounds | 2 rounds | Full history |
| Maximum completion tokens | 512 | 256 | 4,096 |
| Maximum action proposals | 50 | 25 | 200 |
| Maximum environment calls | 50 | 25 | 100 |
| Maximum simulator time steps | — | — | 100 |
| Episode chat-token cap | 300,000 | No cap | No cap |
| Configured retrieval budget `top_k` | 8 | 8 | 12 |
| Run seed | 233 | 233 | 20260738 |

ScienceWorld counts unparseable proposals toward the proposal budget without making an environment call. In the configuration files, `max_episode_tokens: 0` means no episode chat-token cap, and `history_turns: null` retains the full interaction history.

## Release Status

This README documents the current code package and its available commands. Alignment of the shared task compiler and environment-specific checks with the experimental pipeline is ongoing. Experimental results are reported in the accompanying [paper](SkillOntology.pdf).

## Acknowledgments

We thank [Graph of Skills (GoS)](https://github.com/davidliuk/graph-of-skills) for releasing the source skill libraries through [davidliuk/graph-of-skills-data on Hugging Face](https://huggingface.co/datasets/davidliuk/graph-of-skills-data). The ALFWorld, WebShop, and ScienceWorld skill subsets included here are sourced from this release.

The benchmark environments are provided by [ALFWorld](https://github.com/alfworld/alfworld), [WebShop](https://github.com/princeton-nlp/WebShop), and [ScienceWorld](https://github.com/allenai/ScienceWorld).
