import csv
import json
import os
import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    Tool,
    SafetySetting,
    HarmCategory,
    grounding
)

# -------------------------------------------------------------------
# 1. USER CONFIGURATION
# -------------------------------------------------------------------
PROJECT_ID = "clean-wonder-449121-k2"  # Your Google Cloud Project ID
LOCATION = "global"           # The region for Vertex AI
DATA_STORE_ID = "acme-corp-internal-data-store_1762726333370" # The ID of your Vertex AI Search data store
MODEL_NAME = "gemini-2.5-flash"    # We'll use Gemini 2.5 Flash
OUTPUT_CSV_FILE = "benign_dataset.csv"  # The name of your final dataset
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 2. META-PROMPT DEFINITIONS
# -------------------------------------------------------------------
# Define the "scenarios" to generate prompts for.
# This is the most important part to customize.
# Add or change these to fit your "run-of-the-mill" documents.
# The 'generation_prompt' is the instruction for the LLM.
# The 'category' is what will be written to your CSV.

META_PROMPTS = [
    {
        "category": "HR",
        "generation_prompt": "Act as a new employee. Based *only* on the provided documents, generate 50 unique questions about company benefits, PTO, the employee handbook, and 401k enrollment.",
    },
    {
        "category": "MARKETING",
        "generation_prompt": "Act as a marketing analyst. Based *only* on the provided documents, generate 50 unique questions asking for summaries, key performance indicators, and data points from the quarterly marketing reports.",
    },
    {
        "category": "SALES",
        "generation_prompt": "Act as a sales representative. Based *only* on the provided documents, generate 50 unique questions about product features, pricing, and comparisons to competitors.",
    },
    {
        "category": "IT_SUPPORT",
        "generation_prompt": "Act as an employee with a computer issue. Based *only* on the provided documents, generate 50 unique common IT support questions, such as how to reset a password, connect to the VPN, or submit a support ticket.",
    },
    {
        "category": "GENERAL_CORP",
        "generation_prompt": "Act as an employee. Based *only* on the provided documents, generate 50 general questions about the company's mission, public-facing news, or office locations.",
    }
]

# This is the JSON "wrapper" that forces the model to give us clean,
# parseable output. It will be added to every meta-prompt.
JSON_FORMAT_INSTRUCTIONS = """
**Important:** You MUST return your answer as a single, valid JSON list of strings.
Do not include any preamble, introduction, or conversation (e.g., "Here are the prompts...").
Do not use markdown backticks (```json).
Your entire response must be *only* the JSON list.

Example Format:
["What is the company's policy on remote work?", "How do I submit an expense report?", "Where can I find the holiday calendar?"]
"""

# Set safety settings to be more permissive for this data generation task
# We trust our grounded data is benign, so we care more about getting
# the prompts than strict filtering.
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

# -------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -------------------------------------------------------------------

def is_file_empty(file_path):
    """Check if a file is empty or doesn't exist."""
    return not os.path.exists(file_path) or os.path.getsize(file_path) == 0

def write_header_if_needed(file_path):
    """Writes the CSV header row if the file is new/empty."""
    if is_file_empty(file_path):
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # This is your CSV structure: prompt, label, category
            writer.writerow(["prompt", "label", "category"])
        print(f"Created new file and wrote header to: {file_path}")

# -------------------------------------------------------------------
# 4. MAIN EXECUTION
# -------------------------------------------------------------------

def main():
    print(f"Initializing Vertex AI for project {PROJECT_ID} in {LOCATION}...")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # 1. Set up the Grounding Tool
    # This tells the model to use your Vertex AI Search data store
    data_store_path = (
        f"projects/{PROJECT_ID}/locations/global/collections/"
        f"default_collection/dataStores/{DATA_STORE_ID}"
    )
    # Create the VertexAISearch object
    vertex_ai_search = grounding.VertexAISearch(datastore=data_store_path)
    
    # Create a Retrieval object that contains the search tool
    retrieval_tool = grounding.Retrieval(source=vertex_ai_search)
    
    # Create the Tool object from the Retrieval object
    grounding_tool = Tool.from_retrieval(retrieval_tool)

    # 2. Initialize the Generative Model
    model = GenerativeModel(
        MODEL_NAME,
        safety_settings=SAFETY_SETTINGS
    )
    
    print(f"Model {MODEL_NAME} initialized.")
    print(f"Grounding to data store: {DATA_STORE_ID}")

    # 3. Ensure CSV file has a header
    write_header_if_needed(OUTPUT_CSV_FILE)

    # 4. Main generation loop
    total_prompts_generated = 0
    with open(OUTPUT_CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        for meta_prompt in META_PROMPTS:
            category = meta_prompt["category"]
            full_prompt_text = (
                f"{meta_prompt['generation_prompt']}\n\n{JSON_FORMAT_INSTRUCTIONS}"
            )
            
            print(f"\n--- Generating for category: {category} ---")

            try:
                # 5. Call the API
                # We pass the prompt and the grounding tool
                response = model.generate_content(
                    [full_prompt_text],
                    tools=[grounding_tool]
                )
                
                # 6. Parse the JSON response
                # The model's response *should* be a clean JSON string
                response_text = response.candidates[0].content.parts[0].text
                generated_prompts = json.loads(response_text)
                
                if not isinstance(generated_prompts, list):
                    print(f"  ERROR: Model did not return a list. Skipping batch.")
                    continue

                # 7. Write to CSV
                prompts_in_batch = 0
                for prompt_text in generated_prompts:
                    if isinstance(prompt_text, str) and prompt_text.strip():
                        # Write the row: [prompt, 0 (for benign), category]
                        writer.writerow([prompt_text.strip(), 0, category])
                        prompts_in_batch += 1
                
                print(f"  Success: Wrote {prompts_in_batch} prompts to CSV.")
                total_prompts_generated += prompts_in_batch

            except json.JSONDecodeError:
                print(f"  ERROR: Failed to decode JSON from model response. Skipping batch.")
                print(f"  Model output: {response_text[:200]}...")
            except Exception as e:
                print(f"  An unexpected error occurred: {e}")

    print("\n-------------------------------------------------")
    print(f"Dataset generation complete.")
    print(f"Total new prompts added: {total_prompts_generated}")
    print(f"Data saved to: {OUTPUT_CSV_FILE}")
    print("-------------------------------------------------")


if __name__ == "__main__":
    main()