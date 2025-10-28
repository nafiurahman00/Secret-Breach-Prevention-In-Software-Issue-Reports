import os
import time
import pandas as pd
import numpy as np
import re
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
# Make sure you have a .env file with GEMINI_API_KEY="YOUR_API_KEY"
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please create a .env file and add your API key.")

genai.configure(api_key=API_KEY)

# --- You can change the model here if needed ---
MODEL = "gemini-2.5-flash" 

# Initialize the model with safety settings to reduce content filtering
# and generation config to ensure more deterministic output.
model = genai.GenerativeModel(
    MODEL,
    safety_settings={'HARM_CATEGORY_HARASSMENT': 'block_none',
                     'HARM_CATEGORY_HATE_SPEECH': 'block_none',
                     'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'block_none',
                     'HARM_CATEGORY_DANGEROUS_CONTENT': 'block_none'},
    generation_config={"temperature": 0} # Set temperature to 0 for more deterministic output
)

# --- Your Existing Code ---

# Load your test data
try:
    df_test = pd.read_csv("test.csv")
except FileNotFoundError:
    print("Error: 'test.csv' not found. Make sure the file is in the same directory as the script.")
    exit()

def create_context_window(text, target_string, window_size=200):
    """Creates a context window around a target string within a larger text."""
    target_index = text.find(target_string)
    
    if target_index != -1:
        start_index = max(0, target_index - window_size)
        end_index = min(len(text), target_index +
                        len(target_string) + window_size)
        context_window = text[start_index:end_index]
        return context_window
    
    return None # Return None if the target string isn't found

# Apply the context window function
df_test['modified_text'] = df_test.apply(lambda row: create_context_window(
    str(row['text']), str(row['candidate_string'])), axis=1)

# Handle cases where context creation might fail
df_test.dropna(subset=['modified_text'], inplace=True)

# Convert numerical labels to string labels for comparison
df_test['label'] = df_test['label'].replace({0: 'Non-sensitive', 1: 'Secret'})

def generate_zero_shot_test_prompt(data_point):
    """Generates a precise prompt for classification."""
    return f"""
Classify the given candidate string as either "Non-sensitive" or "Secret" based on its role in the provided issue report. 

A "Secret" includes sensitive information such as: 
- API keys and secrets (e.g., `sk_test_ABC123`)  
- Private and secret keys (e.g., private SSH keys, private cryptographic keys)  
- Authentication keys and tokens (e.g., `Bearer <token>`)  
- Database connection strings with credentials (e.g., `mongodb://user:password@host:port`)  
- Passwords, usernames, and any other private information that should not be shared openly.  

A "Non-sensitive" string is not considered secret and can be shared openly. This includes:  
- Public keys of any form (e.g., public SSH keys)  
- Non-sensitive configuration values or identifiers  
- Actual-looking keys that are clearly marked as dummy/test (e.g., with comments like '# dummy key' or variable names like 'test_key')  
- Strings that just look random or patterned but are not actually secrets (e.g., `xyz123`, 'xxxx', `abc123`, `EXAMPLE_KEY`, `token_value`)  
- Strings that are clearly placeholders or redacted text (e.g., 'XXXXXXXX', '[REDACTED]', '[TRUNCATED]')  
- **Obfuscated or masked values (e.g., '****', '****123', 'abc...xyz')**  

These are always considered **"Non-sensitive"**, even if they appear in a sensitive context.  

Carefully consider the context of the string in the provided report. If the string is part of authentication, encryption, or access control, it is likely a "Secret". Otherwise, it is "Non-sensitive". Ensure you pay attention to specific patterns like tokens, passwords, or keys in the string.  

Return only the answer as the corresponding label.

candidate_string: {data_point["candidate_string"]}
issue_report: {data_point["modified_text"]}
label: """.strip()

def generate_test_prompt(data_point):
    """Generates a precise prompt for classification with few-shot examples."""
    return f"""
Classify the given candidate string as either "Non-sensitive" or "Secret" based on its role in the provided issue report. 

A "Secret" includes sensitive information such as: 
- API keys and secrets (e.g., `sk_test_ABC123`)  
- Private and secret keys (e.g., private SSH keys, private cryptographic keys)  
- Authentication keys and tokens (e.g., `Bearer <token>`)  
- Database connection strings with credentials (e.g., `mongodb://user:password@host:port`)  
- Passwords, usernames, and any other private information that should not be shared openly.  

A "Non-sensitive" string is not considered secret and can be shared openly. This includes:  
- Public keys of any form (e.g., public SSH keys)  
- Non-sensitive configuration values or identifiers  
- Actual-looking keys that are clearly marked as dummy/test (e.g., with comments like '# dummy key' or variable names like 'test_key')  
- Strings that just look random or patterned but are not actually secrets (e.g., `xyz123`, 'xxxx', `abc123`, `EXAMPLE_KEY`, `token_value`)  
- Strings that are clearly placeholders or redacted text (e.g., 'XXXXXXXX', '[REDACTED]', '[TRUNCATED]')  
- **Obfuscated or masked values (e.g., '****', '****123', 'abc...xyz')**  

These are always considered **"Non-sensitive"**, even if they appear in a sensitive context.  

Carefully consider the context of the string in the provided report. If the string is part of authentication, encryption, or access control, it is likely a "Secret". Otherwise, it is "Non-sensitive". Ensure you pay attention to specific patterns like tokens, passwords, or keys in the string.  

Here are some examples:

Example 1:
candidate_string: sk_live_51HxYzABC123def456
issue_report: The API key sk_live_51HxYzABC123def456 was accidentally committed to the repository and needs to be revoked immediately.
label: Secret

Example 2:
candidate_string: ****
issue_report: The password field shows **** when the user enters their credentials on the login page.
label: Non-sensitive

Example 3:
candidate_string: mongodb://nafiu:51aXiBR16Fg@localhost:27017
issue_report: Connection failed using mongodb://nafiu:51aXiBR16Fg@localhost:27017 - authentication error occurred.
label: Secret

Example 4:
candidate_string: EXAMPLE_API_KEY
issue_report: Set your API key in the config file. Replace EXAMPLE_API_KEY with your actual key from the dashboard.
label: Non-sensitive

Example 5:
candidate_string: ssh-rsa AAAAB3NzaC1yc2EAAAA...
issue_report: Added public SSH key ssh-rsa AAAAB3NzaC1yc2EAAAA... to the authorized_keys file for deployment access.
label: Non-sensitive

Carefully consider the context of the string in the provided report. If the string is part of authentication, encryption, or access control, it is likely a "Secret". Otherwise, it is "Non-sensitive". Ensure you pay attention to specific patterns like tokens, passwords, or keys in the string.  
Return only the answer as the corresponding label, nothing else.
Now classify the following:

candidate_string: {data_point["candidate_string"]}
issue_report: {data_point["modified_text"]}
label: """.strip()

# Create the prompts
X_test = df_test
X_test['prompt_text'] = X_test.apply(generate_test_prompt, axis=1)


# --- New Code for Prediction and Evaluation ---

import json

# Checkpoint settings
CHECKPOINT_FILE = "predictions_checkpoint.json"
SAVE_INTERVAL = 100  # Save every 10 predictions

# Load existing checkpoint if it exists
start_idx = 0
predictions = []
if os.path.exists(CHECKPOINT_FILE):
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint_data = json.load(f)
            predictions = checkpoint_data.get('predictions', [])
            start_idx = len(predictions)
            print(f"Resuming from checkpoint: {start_idx} predictions already completed.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting fresh.")
        start_idx = 0
        predictions = []

# Use tqdm for a progress bar
total_prompts = len(X_test['prompt_text'])
prompts_list = X_test['prompt_text'].tolist()

for i in tqdm(range(start_idx, total_prompts), desc="Getting Predictions", initial=start_idx, total=total_prompts):
    prompt = prompts_list[i]
    try:
        # Generate content using the model
        response = model.generate_content(prompt)
        
        # Clean the output to get only the label
        # .strip() removes leading/trailing whitespace/newlines
        # .replace("'", "").replace('"', '') removes quotes
        cleaned_response = response.text.strip().replace("'", "").replace('"', '')
        print(f"Model response: '{response.text}' -> Cleaned: '{cleaned_response}'")
        # Basic validation to ensure the output is one of the expected labels
        if cleaned_response not in ["Secret", "Non-sensitive"]:
            # If the model gives an unexpected response, default to Non-sensitive
            # to avoid errors in metrics calculation. You could also log this.
            predictions.append("Non-sensitive") 
            print(f"Warning: Unexpected model output: '{cleaned_response}'. Defaulting to 'Non-sensitive'.")
        else:
            predictions.append(cleaned_response)

    except Exception as e:
        # If an API error occurs, append a default value and print the error
        print(f"An error occurred: {e}")
        predictions.append("Non-sensitive") # Default value on error

    # Save checkpoint at intervals
    if (i + 1) % SAVE_INTERVAL == 0 or (i + 1) == total_prompts:
        checkpoint_data = {
            'predictions': predictions,
            'completed_count': len(predictions),
            'total_count': total_prompts
        }
        try:
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(checkpoint_data, f)
            print(f"Checkpoint saved: {len(predictions)}/{total_prompts} predictions completed.")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    # --- Rate Limiting ---
    # Pause for 1 second between each request to stay within the
    # typical free-tier limit of 60 requests per minute.
    time.sleep(1)

# Add predictions to the DataFrame for review
df_test['predicted_label'] = predictions

# # Clean up checkpoint file after successful completion
# if os.path.exists(CHECKPOINT_FILE):
#     try:
#         os.remove(CHECKPOINT_FILE)
#         print("Checkpoint file cleaned up after successful completion.")
#     except Exception as e:
#         print(f"Note: Could not remove checkpoint file: {e}")

# --- Calculate and Display Metrics ---

# Get the actual and predicted labels
true_labels = df_test['label']
predicted_labels = df_test['predicted_label']

# Calculate precision, recall, and f1-score
precision = precision_score(true_labels, predicted_labels, pos_label='Secret')
recall = recall_score(true_labels, predicted_labels, pos_label='Secret')
f1 = f1_score(true_labels, predicted_labels, pos_label='Secret')

print("\n--- Individual Metrics (for 'Secret' class) ---")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Display a comprehensive classification report
print("\n--- Full Classification Report ---")
print(classification_report(true_labels, predicted_labels, digits=4))

# Save the results to a new CSV file for detailed analysis
df_test.to_csv("test_with_predictions.csv", index=False)
print("\nResults saved to 'test_with_predictions.csv'")