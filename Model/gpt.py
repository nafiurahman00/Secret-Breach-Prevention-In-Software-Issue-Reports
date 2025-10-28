import os
import time
import pandas as pd
import numpy as np
import re
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Configure OpenAI API
# Make sure you have a .env file with OPENAI_API_KEY="YOUR_API_KEY"
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please create a .env file and add your API key.")

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

MODEL = "gpt-4o" 

# --- Your Existing Code ---

# Output CSV file with predictions
OUTPUT_CSV = "test_with_predictions_openai.csv"

# Load your test data or resume from existing output
if os.path.exists(OUTPUT_CSV):
    print(f"Found existing output file '{OUTPUT_CSV}'. Resuming from where we left off...")
    df_test = pd.read_csv(OUTPUT_CSV)
    # Ensure predicted_label column exists
    if 'predicted_label' not in df_test.columns:
        df_test['predicted_label'] = np.nan
else:
    print(f"Loading fresh data from 'test.csv'...")
    try:
        df_test = pd.read_csv("test.csv")
        # Add predicted_label column, initialized as NaN
        df_test['predicted_label'] = np.nan
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
from openai import RateLimitError, APIError

# Save interval - save progress every 10 predictions to minimize data loss
SAVE_INTERVAL = 10

# Rate limit handling configuration
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1  # seconds
MAX_RETRY_DELAY = 60  # seconds

# Use tqdm for a progress bar
total_prompts = len(X_test)
prompts_list = X_test['prompt_text'].tolist()

# Count how many predictions are already done
already_completed = df_test['predicted_label'].notna().sum()
remaining = total_prompts - already_completed

print(f"\n=== Progress Status ===")
print(f"Total rows: {total_prompts}")
print(f"Already completed: {already_completed}")
print(f"Remaining: {remaining}")
print(f"Output file: {OUTPUT_CSV}")
print("=" * 50 + "\n")

if remaining == 0:
    print("All predictions are already complete!")
else:
    # Process only rows with empty predicted_label
    api_exhausted = False
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 5  # Stop after 5 consecutive API errors (likely budget exhausted)
    
    for idx in tqdm(range(len(df_test)), desc="Getting Predictions", total=total_prompts):
        # Skip if predicted_label is already filled
        if pd.notna(df_test.loc[idx, 'predicted_label']):
            continue
        
        if api_exhausted:
            break
            
        prompt = df_test.loc[idx, 'prompt_text']
        
        # Retry logic with exponential backoff for rate limits
        retry_count = 0
        retry_delay = INITIAL_RETRY_DELAY
        success = False
        
        while retry_count < MAX_RETRIES and not success:
            try:
                # Generate content using OpenAI API
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a precise classifier that returns only 'Secret' or 'Non-sensitive'."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,  # Set to 0 for more deterministic output
                    max_tokens=4   # We only need a short response
                )
                
                # Extract the response text
                response_text = response.choices[0].message.content
                
                # Clean the output to get only the label
                cleaned_response = response_text.strip().replace("'", "").replace('"', '')
                print(f"\nRow {idx} - Model response: '{response_text}' | Cleaned: '{cleaned_response}'")
                
                # Basic validation to ensure the output is one of the expected labels
                if cleaned_response not in ["Secret", "Non-sensitive"]:
                    # If the model gives an unexpected response, default to Non-sensitive
                    df_test.loc[idx, 'predicted_label'] = "Non-sensitive"
                    print(f"Warning: Unexpected model output: '{cleaned_response}'. Defaulting to 'Non-sensitive'.")
                else:
                    df_test.loc[idx, 'predicted_label'] = cleaned_response
                
                # Reset consecutive error counter on success
                consecutive_errors = 0
                success = True

            except RateLimitError as e:
                retry_count += 1
                error_msg = str(e).lower()
                
                # Check if it's a quota exhaustion (not recoverable) or rate limit (recoverable)
                if 'insufficient_quota' in error_msg or 'exceeded your current quota' in error_msg:
                    print(f"\n{'='*50}")
                    print(f"API QUOTA EXHAUSTED!")
                    print(f"Error: {e}")
                    print(f"{'='*50}")
                    print(f"Saving progress and stopping...")
                    api_exhausted = True
                    break
                else:
                    # Rate limit hit - retry with exponential backoff
                    if retry_count < MAX_RETRIES:
                        print(f"\n⚠️  Rate limit hit! Retry {retry_count}/{MAX_RETRIES}")
                        print(f"   Waiting {retry_delay} seconds before retrying...")
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)  # Exponential backoff
                    else:
                        print(f"\n{'='*50}")
                        print(f"MAX RETRIES REACHED - Rate limit persists")
                        print(f"Error: {e}")
                        print(f"{'='*50}")
                        print(f"Saving progress and stopping...")
                        api_exhausted = True
                        break
                        
            except APIError as e:
                error_msg = str(e).lower()
                
                # Check for specific API errors
                if any(keyword in error_msg for keyword in ['quota', 'insufficient', 'billing', 'exceeded']):
                    print(f"\n{'='*50}")
                    print(f"API BILLING/QUOTA ERROR!")
                    print(f"Error: {e}")
                    print(f"{'='*50}")
                    print(f"Saving progress and stopping...")
                    api_exhausted = True
                    break
                else:
                    consecutive_errors += 1
                    print(f"\nAPI Error occurred: {e}")
                    print(f"Consecutive errors: {consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}")
                    
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        print(f"\n{'='*50}")
                        print(f"TOO MANY CONSECUTIVE API ERRORS")
                        print(f"Saving progress and stopping...")
                        print(f"{'='*50}")
                        api_exhausted = True
                        break
                    
                    # Short delay before next iteration
                    time.sleep(1)
                    break  # Exit retry loop and move to next item

            except Exception as e:
                consecutive_errors += 1
                print(f"\nUnexpected error occurred: {e}")
                print(f"Consecutive errors: {consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}")
                
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print(f"\n{'='*50}")
                    print(f"TOO MANY CONSECUTIVE ERRORS")
                    print(f"Saving progress and stopping...")
                    print(f"{'='*50}")
                    api_exhausted = True
                    break
                
                # Short delay before next iteration
                time.sleep(1)
                break  # Exit retry loop and move to next item

        # Save progress at intervals
        if (idx + 1) % SAVE_INTERVAL == 0:
            try:
                df_test.to_csv(OUTPUT_CSV, index=False)
                completed = df_test['predicted_label'].notna().sum()
                print(f"\nProgress saved: {completed}/{total_prompts} predictions completed.")
            except Exception as e:
                print(f"\nError saving progress: {e}")

        # --- Rate Limiting ---
        # Adaptive delay based on model to avoid hitting rate limits
        # gpt-4o has stricter rate limits than gpt-4o-mini
        if "gpt-4o-mini" in MODEL:
            time.sleep(0.1)  # Shorter delay for mini model
        else:
            time.sleep(0.5)  # Longer delay for full gpt-4o model

    # Final save after loop ends
    try:
        df_test.to_csv(OUTPUT_CSV, index=False)
        completed = df_test['predicted_label'].notna().sum()
        print(f"\n{'='*50}")
        print(f"Final save completed: {completed}/{total_prompts} predictions done.")
        print(f"Results saved to '{OUTPUT_CSV}'")
        print(f"{'='*50}\n")
    except Exception as e:
        print(f"\nError in final save: {e}")

# --- Calculate and Display Metrics ---

# Only calculate metrics if we have completed predictions
completed_count = df_test['predicted_label'].notna().sum()

if completed_count == len(df_test):
    print("\n" + "="*50)
    print("ALL PREDICTIONS COMPLETED!")
    print("="*50)
    
    # Get the actual and predicted labels (only for rows with predictions)
    mask = df_test['predicted_label'].notna()
    true_labels = df_test.loc[mask, 'label']
    predicted_labels = df_test.loc[mask, 'predicted_label']

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
    
    print(f"\nFinal results saved to '{OUTPUT_CSV}'")
else:
    print(f"\n{'='*50}")
    print(f"PARTIAL COMPLETION")
    print(f"{'='*50}")
    print(f"Completed: {completed_count}/{len(df_test)}")
    print(f"Remaining: {len(df_test) - completed_count}")
    print(f"\nTo resume, simply run this script again.")
    print(f"Progress has been saved to '{OUTPUT_CSV}'")
    
    # Calculate metrics for completed rows only (if any)
    if completed_count > 0:
        mask = df_test['predicted_label'].notna()
        true_labels = df_test.loc[mask, 'label']
        predicted_labels = df_test.loc[mask, 'predicted_label']
        
        try:
            precision = precision_score(true_labels, predicted_labels, pos_label='Secret')
            recall = recall_score(true_labels, predicted_labels, pos_label='Secret')
            f1 = f1_score(true_labels, predicted_labels, pos_label='Secret')
            
            print("\n--- Metrics for Completed Predictions So Far ---")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
        except Exception as e:
            print(f"\nNote: Could not calculate metrics yet: {e}")
