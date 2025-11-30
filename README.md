***

# NLP Firewall: A Lightweight DLP Classifier for RAG-Enabled LLM Prompting

## Project Overview

This project is a proof-of-concept for a Case Studies in Machine Learning course to determine the feasibility of a lightweight "LLM Firewall." The goal: create a fast, efficient, and lightweight Data Loss Prevention (DLP) model that classifies user prompts before they reach an internal Large Language Model (LLM) equipped with Retrieval-Augmented Generation (RAG).

Organizations often struggle to balance data security with LLMs, especially when leveraging internal data for context. Most DLP solutions rely on large, multi-billion parameter models that are costly and slow, making them impractical for real-time applications and out-of-reach for some organizations due to infrastructure constraints.

### Core Hypothesis

A small, fine-tuned transformer model (like DistilBERT) can achieve high recall, precision, and F1—while being significantly more efficient (lower latency)—compared to a large, multi-billion parameter model (like [meta-llama/Llama-Guard-3-8b](https://huggingface.co/meta-llama/Llama-Guard-3-8B)) for this narrow DLP task.

***

## Project Structure

The project includes scripts for data generation, model training, and model evaluation.

### Key Files & Directories

**Data Files**
- `benign_dataset.csv`: ~9,000 "benign" corporate prompts (generated).
- `malicious_dataset.csv`: ~1,000 manually created "malicious" exfiltration prompts.
- `full_dataset.csv`: Combined, shuffled dataset of ~10,000 prompts.

**Model Training Scripts**
- `generate_dataset.py`: Uses Vertex AI Search (grounded on non-sensitive corporate docs) to generate realistic, safe prompts.
- `NLP_Fine_Tune.py`: Loads `full_dataset.csv`, splits 80/20, fine-tunes DistilBERT.

**Model Evaluation Scripts**
- `evaluate_models.py`: Compares the local model to Llama Guard, printing a summary table.
- `evaluate_models_by_category.py`: Offers detailed metrics (Precision, Recall, F1) per prompt category.
- `generate_visualizations.py`: Runs a full evaluation and generates/saves:
    - Overall metrics and per-category F1 tables (Markdown).
    - `confusion_matrix_local.png`
    - `confusion_matrix_llama_guard.png`
    - `roc_curve_local.png` (for DistilBERT)
    - `latency_comparison.png`

**Model Artifacts**
- `my_final_firewall_model/`: The saved, fine-tuned DistilBERT model.

***

## Methodology & Pipeline

The research pipeline follows these core steps:

1. **Data Generation**  
   - Benign prompts: Created by `generate_dataset.py` using a grounded Gemini model on Vertex AI Search.  
   - Malicious prompts: Manually curated.  
   - Combined and shuffled into `full_dataset.csv`.
2. **Model Training**
   - `NLP_Fine_Tune.py` trains and saves the DistilBERT model.
3. **Model Evaluation**
   - `generate_visualizations.py` runs evaluation on the holdout set against both DistilBERT and Llama Guard 2.
4. **Result Generation**
   - Metrics (Accuracy, Precision, Recall, F1, Latency) are reported. Final tables and plots are auto-generated.

***

## Environment Setup

It is recommended to use conda as your virtual environment manager. However, PyTorch does **not** support conda distributions. **Install PyTorch via pip within your activated conda environment.**

For faster training and evaluation, use PyTorch with GPU support. See the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) for details.

### Install Project Dependencies

> **requirements.txt is provided.** Run the following commands to install required packages:

```sh
pip install -r requirements.txt
```

### Hardware Requirements

- **DistilBERT fine-tuning** and **Llama Guard 8B evaluation** are resource-intensive.
- Llama Guard 8B inference requires **~16 GB GPU memory**.
- For optimal performance, use a machine with at least 16GB of GPU RAM.
- All experiments were performed with an NVIDIA RTX 3090 GPU.

***

## Usage / Reproducing Results

Follow these steps from your activated `llm_firewall` environment:

### Step 1: Generate & Prepare Data

- Configure `generate_dataset.py` with your **GCP Project ID** and **Vertex AI Search Data Store ID**.
    - Example: "dummy" corporate documents (in `acme-sample-corpus`) grounded in a Vertex AI Search data store.

- To create the **benign** dataset:
    ```sh
    python BenignDataGen_FULL.py
    ```
- To create the **malicious** dataset:
    ```sh
    python MaliciousDataGen.py
    ```

### Step 2: Train Your Models

- Fine-tune DistilBERT (saves model in `my_final_firewall_model/`):
    ```sh
    python NLP_Fine_Tune.py
    ```

### Step 3: Run Evaluation & Generate Plots

- Perform full evaluation and generate plots/tables:
    ```sh
    python evaluation.py
    python metric_viz.py
    ```

- Output: Results tables (printed to console), `.png` plots saved to project directory.

#### (Optional) Monitor Training in Real Time

- Open a second terminal tab/window:
    ```sh
    # In Terminal 1
    python NLP_Fine_Tune.py

    # In Terminal 2
    tensorboard --logdir my_llm_firewall_model
    ```

***

## Find the Full Report under 'CSML Final Report.pdf'"