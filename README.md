# 📊 Highly-Biased

This repository contains work and code for the paper **"A case study on Positional Bias in Large Language Models"**. 

## 📝 Introduction

Large Language Models (LLMs) are widely used for a variety of tasks such as text generation, ranking, and decision-making.
However, their outputs can be influenced by various forms of biases.
One such bias is positional bias, where models prioritize items based on their position within a given prompt rather than their content or quality, impacting on how LLMs interpret and weigh information, potentially compromising fairness, reliability, and robustness.
To assess positional bias, we prompt a range of  LLMs to generate descriptions for a list of topics, systematically permuting their order and analyzing variations in the responses.
Our analysis shows that ranking position affects structural features and coherence, with some LLMs also altering or omitting topic order.
Nonetheless, the impact of positional bias varies across different LLMs and topics, indicating an interplay with other related biases.

## 🛠️ Project Setup

This project uses **Poetry** for dependency management.

### 🚀 Steps to Set Up

1. 📦 **Install Poetry** (if not already installed):

   Using `pipx` (recommended):
   ```bash
    pipx install poetry
   ```

   Using `pip`:
      ```bash
    pip install poetry
   ```

   For more installation options, refer to [Poetry’s documentation](https://python-poetry.org/docs/).

2. 📥 **Install dependencies:** Navigate to the project directory and install the required packages:
    ```bash
   poetry install
   ```

3. 🪄 **Activate the environment:**
    ```bash
   poetry shell
   ```

## 📃 Repository Structure

```
📂 Highly-Biased
├── 📄 create_update_dataset.py
├── 📄 openai_batch_api.ipynb
├── 📄 poetry.lock
├── 📄 pyproject.toml
├── 📄 statistical_tests_and_plot_generation.ipynb
├── 📄 rouge_metric_multithreading.py
├── 📄 run_evaluation.py
├── 📄 run_generate_golden_descriptions.py
├── 📄 run_gpt_evaluation.py
├── 📄 visualize.ipynb
├── 📁 charts
├── 📁 datasets
├── 📁 helper
├── 📁 models
├── 📁 openai-batch-files
├── 📁 plots
│   ├── 📊 boxplots
│   ├── 🌡️ heatmaps
│   ├── 🐝 swarmplots
└── 📁 scorer
```

### 🔑 **Key Components**:
- 📁 `charts`: Description
- 📁 `datasets`: Description
- 🤖 `models`:  Description 
- `openai-batch-files`:  Description
- 📈 `plots`:  
  - 📊 `boxplots`:  
    - `joined_boxplots_per_position`: Joined across five models at a time, displaying the distribution of metrics across positions.
    - `joined_boxplots_per_topic`: Joined across five models at a time, displaying the distribution of metrics across topics.   
    - `per_position`: Boxplots displaying the distribution of metrics across positions.
    - `per_position_axis_fixed`: Boxplots displaying the distribution of metrics across positions, with a fixed y-axis across each metric. 
    - `per_topic`: Boxplots displaying the distribution of metrics across topics. 
    - `per_topic_axis_fixed`: Boxplots displaying the distribution of metrics across positions, with a fixed y-axis across each metric. 
  - 🌡️ `heatmaps`:  
    - `per_position`: Heatmaps showing the correlation between different position.
    - `per_topics`: Heatmaps showing the correlation between different topics.  
  - 🐝 `swarmplots`:  
    - `per_position`: Swarmplots displaying the distribution of metrics across positions.  
    - `per_position_axis_fixed`: Swarmplots displaying the distribution of metrics across positions, with a fixed y-axis across each metric.
- 🦾 `scorer`: Description  
- 📄 `create_swarmplot.py`: Description  
- 📄 `create_update_dataset.py`: Description  
- 📄 `openai_batch_api.ipynb`: Description  
- 🧩 `poetry.lock` and `pyproject.toml`: Poetry files for dependency management.
- 📊`statistical_tests_and_plot_generation.ipynb`: Contains statistical tests and code for generating boxplots, heatmaps, swarmplots, and bar charts for Missing Topic or Position.
- 📄 `rouge_metric_multithreading.py`: Description  
- 📄 `run_evaluation.py`: Description  
- 📄 `run_generate_golden_descriptions.py`: Description  
- 📄 `run_gpt_evaluation.py`: Description  
- 📄 `run_reevaluation.py`: Description  
- 📊 `visualize.ipynb`: Description
- 📁 `helper`:
  - 📄 `create_swarmplot.py`: Helper file for creating swarmplots
  - 📄 `batch_loading.py`: Description

