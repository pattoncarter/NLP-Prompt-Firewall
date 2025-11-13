# NLP Firewall: A Lightweight DLP Classifier for RAG Enabled LLM Prompting

1. Project Overview

This project is a proof-of-concept for my Case Studies in Machine Learning course to determine the feasibility of a lightweight "LLM Firewall." The goal is to create a fast, efficient, and lightweight Data Loss Prevention (DLP) model that can classify user prompts before they reach an internal Large Language Model (LLM) equipped with Retrieval-Augmented Generation (RAG).

Through my work as a cybersecurity consultant, I've observed that organizations often struggle to balance data security with the power of LLMs, especially when leveraging internal data as a context resource. Many existing solutions rely on large, multi-billion parameter models that are costly and slow, making them impractical for real-time applications, and infeasible for some organizations due to budget or infrastructure constraints.

Core Hypothesis

This research tests the hypothesis that a small, fine-tuned transformer model (like DistilBERT) can be as effective (high recall, precision, F1) and significantly more efficient (low latency) than a large, multi-billion parameter model (like meta-llama/Llama-Guard-3-8b) for this "narrow" classification task.

2. Project Structure

This project consists of data generation, model training, and model evaluation scripts.

Key Files & Directories

Data Files

benign_dataset.csv: (Generated) A CSV of ~9,000 "benign" corporate prompts.

malicious_dataset.csv: (Manually created) A CSV of ~1,000 "malicious" exfiltration prompts.

full_dataset.csv: The final combined and shuffled dataset of ~10,000 prompts.

Model Training Scripts

generate_dataset.py: (From conversation history) Generates benign_dataset.csv by using Vertex AI Search (grounded on non-sensitive corporate docs) to create realistic, safe prompts.

NLP_Fine_Tune.py: (From conversation history) The main script. It loads full_dataset.csv, splits it 80/20, and fine-tunes a DistilBERT model.

Model Evaluation Scripts

evaluate_models.py: A simple evaluation script that compares the local model vs. Llama Guard and prints a results table.

evaluate_models_by_category.py: A detailed evaluation script that provides a performance breakdown (Precision, Recall, F1) for each prompt category.

generate_visualizations.py: The primary script for the paper's results. It runs a full evaluation and generates/saves:

Overall metrics table (Markdown).

Per-category F1-score table (Markdown).

confusion_matrix_local.png

confusion_matrix_llama_guard.png

roc_curve_local.png (for the DistilBERT model)

latency_comparison.png

Model Artifacts

my_final_firewall_model/: (Generated) The saved, fine-tuned DistilBERT model.

3. Methodology & Pipeline

The project follows a 4-step process to replicate the research.

Data Generation: The benign dataset is created by generate_dataset.py, which uses a grounded Gemini model on a Vertex AI Search data store to produce realistic corporate-style prompts. The malicious dataset is created manually. These are combined into full_dataset.csv.

Model Training:

The NLP_Fine_Tune.py script trains the primary DistilBERT model and saves it.

Model Evaluation: The generate_visualizations.py script loads the validation split (the 20% hold-out set) and runs it against both our fine-tuned DistilBERT and the 8B-parameter Llama Guard 2 model.

Result Generation: The evaluation script measures all key metrics (Accuracy, Precision, Recall, F1, Latency) and generates the final tables and plots for the paper.

4. Environment Setup

The enviornment can be set up using conda as a virtual environment manager (as I have); however, it should be noted that PyTorch does not support conda distributions. Therefore, it is recommended to install PyTorch via pip within the conda environment.

It is recommended to leverage PyTorch with GPU support for faster training & evaluation times. Please refer to the official PyTorch installation guide for instructions on installing the appropriate version for your system: https://pytorch.org/get-started/locally/

Install Project Dependencies:
A requirements.txt is not provided, but you can install all required packages by running:

pip install transformers datasets evaluate pandas scikit-learn torch
pip install matplotlib seaborn jupyter
pip install accelerate
pip install google-cloud-aiplatform
pip install tokenizers
pip install tensorboard

### Hardware Requirements:
It shoiuld be noted that the DistilBERT fine-tuning process & evaluation of llama-guard-3-8b can be resource-intensive. During testing, I found that the llama-guard-3-8b model required ~16GB of GPU memory to run inference effectively. Therefore, it is recommended to use a machine with at least 16GB of GPU memory for optimal performance. I performed my experiments with an NVIDIA RTX 3090 GPU.

5. Usage / How to Replicate Results

Follow these steps in order from your activated llm_firewall environment.

Step 1: Generate & Prepare Data

Configure generate_dataset.py with your GCP Project ID and Vertex AI Search Data Store ID. My GCP project consisted of non-sensitive, "dummy" corporate documentation (found in acme-sample-corpus) to ground the benign prompt generation stored in a Vertex AI Search data store.

Run the script to create benign_dataset.csv:

python BenignDataGen_FULL.py


Run the script to create malicious_dataset.csv:

python MaliciousDataGen.py

Step 2: Train Your Models

Run the DistilBERT fine-tuning script. This will create the my_final_firewall_model directory.

python NLP_Fine_Tune.py


Step 3: Run Evaluation & Generate Plots

Run the main evaluation & evaluation scripts. This will perform the full evaluation against your model and Llama Guard, then save the .png plots and print your results tables to the console.

python evaluation.py

python metric_viz.py



(Optional) To watch your training run (for NLP_Fine_Tune.py) in real-time, use TensorBoard.

Terminal 1: python NLP_Fine_Tune.py

Terminal 2: tensorboard --logdir my_llm_firewall_model