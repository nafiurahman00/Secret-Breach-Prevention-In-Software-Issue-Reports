import requests
import time
import re
import pandas as pd
from tqdm import tqdm
import csv
import os

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
MAX_RESULTS = 1000
PER_PAGE = 100
OUTPUT_CSV = "filtered_github_secrets_2.csv"
REGEX_FILE = "Secret-Regular-Expression.xlsx"
SAVE_EVERY = 1000  
SKIP_URLS_FILE = "alltp.txt"

# Stricter keywords that avoid placeholders and focus on actual secrets
SEARCH_KEYWORDS = [
    # === Bearer tokens (most common format) ===
    "Bearer ey", "Bearer gh", "Bearer sk-", "Bearer pk-", "Bearer xoxb-", "Bearer xoxp-",
    "Bearer AIza", "Bearer ya29", "Bearer AKIA", "Bearer ghp_", "Bearer ghs_",
    "Bearer github_pat_", "Bearer glpat-", "Bearer gho_", "Bearer ghu_",
    
    # === Actual GitHub tokens ===
    '"ghp_', "'ghp_", '"ghs_', "'ghs_", '"github_pat_', "'github_pat_",
    "ghp_1", "ghp_2", "ghs_1", "ghs_2", "github_pat_1", "github_pat_2",
    "token=ghp_", "TOKEN=ghp_", "github_token=ghp_",
    
    # === AWS patterns ===
    '"AKIA', "'AKIA", "AKIA1", "AKIA2", "AKIAI", "AKIAJ", "AKIAK",
    "aws_access_key_id=AKIA", "AWS_ACCESS_KEY_ID=AKIA",
    "AccessKeyId=AKIA", "access_key=AKIA",
    
    # === OpenAI API keys ===
    '"sk-', "'sk-", "sk-1", "sk-2", "sk-3", "sk-proj-", "sk-test-",
    "openai_api_key=sk-", "OPENAI_API_KEY=sk-", "api_key=sk-",
    "apikey=sk-", "secret_key=sk-",
    
    # === Slack tokens ===
    '"xoxb-', "'xoxb-", '"xoxp-', "'xoxp-", "xoxb-1", "xoxb-2", "xoxp-1",
    "slack_token=xoxb-", "SLACK_TOKEN=xoxb-", "bot_token=xoxb-",
    
    # === Stripe keys ===
    '"sk_live_', "'sk_live_", '"sk_test_', "'sk_test_", "sk_live_1", "sk_test_1",
    '"pk_live_', "'pk_live_", '"pk_test_', "'pk_test_", "pk_live_1", "pk_test_1",
    "stripe_key=sk_", "STRIPE_KEY=sk_", "api_key=sk_live_",
    
    # === Google API keys ===
    '"AIza', "'AIza", "AIza1", "AIza2", "AIzaS", "AIzaG",
    '"ya29.', "'ya29.", "ya29.a", "ya29.c", "ya29.1",
    "google_api_key=AIza", "GOOGLE_API_KEY=AIza", "api_key=AIza",
    
    # === Discord tokens ===
    '"eyJ', "'eyJ", "eyJhbGci", "eyJ0eXAi", "mfa.1", "mfa.a",
    "discord_token=", "DISCORD_TOKEN=", "bot_token=eyJ",
    
    # === JWT tokens (with actual structure) ===
    '"eyJ', "'eyJ", "eyJhbGci", "eyJ0eXAi", "eyJraWQi", "eyJpc3Mi",
    "jwt=eyJ", "token=eyJ", "access_token=eyJ", "bearer eyJ",
    
    # === Database connection strings (actual ones) ===
    "mongodb://user:", "mysql://root:", "postgresql://admin:", "redis://default:",
    "mongodb+srv://", "postgres://user:", "mysql://admin:", "redis://user:",
    "://localhost:", "://127.0.0.1:", "://db:", "://database:",
    
    # === Private keys (actual PEM format) ===
    "-----BEGIN RSA PRIVATE KEY-----",
    "-----BEGIN PRIVATE KEY-----",
    "-----BEGIN OPENSSH PRIVATE KEY-----",
    "-----BEGIN EC PRIVATE KEY-----",
    "-----BEGIN DSA PRIVATE KEY-----",
    
    # === API keys with specific patterns ===
    "api_key=", "apikey=", "secret=", "token=", "key=", "password=",
    "API_KEY=", "SECRET_KEY=", "ACCESS_TOKEN=", "CLIENT_SECRET=",
    '"api_key":', '"secret":', '"token":', '"password":', '"key":',
    
    # === Environment variables with actual values ===
    "API_KEY=", "SECRET_KEY=", "ACCESS_TOKEN=", "CLIENT_SECRET=",
    "GITHUB_TOKEN=gh", "OPENAI_API_KEY=sk-", "AWS_ACCESS_KEY=AKIA",
    "export API_KEY=", "export SECRET=", "export TOKEN=",
    
    # === Telegram bot tokens ===
    "telegram_token=", "bot_token=", "TELEGRAM_TOKEN=", "BOT_TOKEN=",
    "telegram_api=", "bot_api=", '"bot_token":', '"telegram_token":',
    
    # === Heroku API keys ===
    "heroku_api_key=", "HEROKU_API_KEY=", "heroku_token=",
    '"heroku_api_key":', '"heroku_token":', "api_key=heroku",
    
    # === Azure secrets ===
    "azure_client_secret=", "AZURE_CLIENT_SECRET=", "azure_key=",
    "azure_storage_key=", "AZURE_STORAGE_KEY=", "azure_token=",
    
    # === Firebase ===
    "firebase_api_key=AIza", "FIREBASE_API_KEY=AIza", "firebase_key=",
    "firebase-adminsdk", "service_account_key", "google-services.json",
    
    # === SendGrid ===
    '"SG.', "'SG.", "sendgrid_api_key=SG", "SENDGRID_API_KEY=SG",
    "SG.1", "SG.2", "SG.A", "sendgrid_key=",
    
    # === Twilio ===
    "twilio_auth_token=", "TWILIO_AUTH_TOKEN=", "twilio_sid=",
    "twilio_account_sid=AC", "TWILIO_ACCOUNT_SID=AC", "AC1", "AC2", "SK1", "SK2",
    
    # === DigitalOcean ===
    "do_api_token=dop_v1_", "DO_API_TOKEN=dop_v1_", "digitalocean_token=",
    "dop_v1_", "do_token=", "digital_ocean_key=",
    
    # === Cloudflare ===
    "cloudflare_api_key=", "CLOUDFLARE_API_KEY=", "cloudflare_token=",
    "CLOUDFLARE_API_TOKEN=", "cf_api_key=", "cf_token=",
    
    # === Actual secret patterns (Base64 and hex encoded) ===
    "==" , "===", "==}", '=="', "=='", "base64:",
    "encoding=", "encoded=", "hash=", "digest=",
    
    # === Header authorization patterns ===
    "Authorization: Bearer", "Authorization: Basic", "Authorization:",
    "X-API-Key:", "X-Auth-Token:", "X-Access-Token:", "Bearer ",
    
    # === Configuration with actual values ===
    "client_id=", "client_secret=", "access_token=", "refresh_token=",
    "CLIENT_ID=", "CLIENT_SECRET=", "ACCESS_TOKEN=", "REFRESH_TOKEN=",
    
    # === JSON/YAML patterns with actual values ===
    '"secret":', '"api_key":', '"token":', '"password":', '"key":',
    "'secret':", "'api_key':", "'token':", "'password':", "'key':",
    
    # === Connection strings with credentials ===
    "mongodb+srv://", "postgres://", "mysql://", "://user:", "://admin:",
    "://root:", "://db:", "://localhost:", "://127.0.0.1:",
    
    # === Kubernetes secrets ===
    "kubectl", "secret", "--from-literal", "configmap", "base64",
    
    # === SSH keys ===
    "ssh-rsa AAAAB3NzaC1yc2", "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5",
    "ssh-dss AAAAB3NzaC1kc3M", "-----BEGIN OPENSSH PRIVATE KEY-----",
    
    # === Certificate patterns ===
    "-----BEGIN CERTIFICATE-----", "-----END CERTIFICATE-----",
    "-----BEGIN RSA PRIVATE KEY-----", "-----END RSA PRIVATE KEY-----",
    
    # === Docker secrets ===
    "DOCKER_PASSWORD=", "DOCKERHUB_TOKEN=", "DOCKER_TOKEN=",
    "docker_password=", "dockerhub_token=", "registry_password=",
]

def load_regex_patterns(file_path):
    df = pd.read_excel(file_path)
    regex_list = []
    for _, row in df.iterrows():
        if pd.notna(row['Regular Expression']):
            try:
                regex_list.append((row['Secret Type'], re.compile(row['Regular Expression'], re.IGNORECASE)))
            except re.error as e:
                print(f"⚠️ Invalid regex skipped: {row['Regular Expression']} ({e})")
    return regex_list

def search_issues(keyword, page):
    url = f"https://api.github.com/search/issues?q={keyword}+in:body+type:issue&per_page={PER_PAGE}&page={page}"
    response = requests.get(url, headers=HEADERS)

    if response.status_code == 403 and "X-RateLimit-Reset" in response.headers:
        reset_time = int(response.headers["X-RateLimit-Reset"])
        wait_seconds = max(0, reset_time - int(time.time()) + 5)
        print(f"⏳ Rate limit hit. Sleeping for {wait_seconds}s...")
        time.sleep(wait_seconds)
        return search_issues(keyword, page)

    response.raise_for_status()
    return response.json()["items"]

def write_partial_results(partial_results, write_header):
    df = pd.DataFrame(partial_results)
    df.to_csv(OUTPUT_CSV, mode='a', index=False, header=write_header, encoding="utf-8")

def crawl():
    regex_patterns = load_regex_patterns(REGEX_FILE)
    seen_urls = set()
    partial_results = []
    header_written = os.path.exists(OUTPUT_CSV)
    skip_urls = set()
    if os.path.exists(SKIP_URLS_FILE):
        with open(SKIP_URLS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                url = line.strip()
                if url:
                    skip_urls.add(url)
            print(f"📋 Loaded {len(skip_urls)} URLs to skip from {SKIP_URLS_FILE}")
            
    if not header_written:
        open(OUTPUT_CSV, 'w').close()  # clear file if new

    print(f"🔍 Searching using {len(SEARCH_KEYWORDS)} strict keywords and applying {len(regex_patterns)} regex patterns...")
    print("🎯 Focusing on actual secrets, avoiding placeholders like YOUR_SECRET_TOKEN")

    for keyword in SEARCH_KEYWORDS:
        print(f"\n📌 Searching for keyword: '{keyword}'")
        for page in tqdm(range(1, MAX_RESULTS // PER_PAGE + 1)):
            try:
                issues = search_issues(keyword, page)
                if not issues:
                    break

                for issue in issues:
                    url = issue["html_url"]
                    if url in seen_urls or url in skip_urls:
                        continue

                    body = issue.get("body", "") or ""
                    
                    # Additional filtering to avoid common placeholders
                    if any(placeholder in body.upper() for placeholder in [
                        "YOUR_SECRET", "YOUR_API_KEY", "YOUR_TOKEN", "REPLACE_ME", 
                        "PLACEHOLDER", "EXAMPLE_", "TEST_SECRET", "DUMMY_", 
                        "FAKE_", "SAMPLE_", "<YOUR_", "ADD_YOUR_", "INSERT_YOUR_",
                        "PUT_YOUR_", "ENTER_YOUR_", "CHANGE_THIS", "REPLACE_THIS"
                    ]):
                        continue
                    
                    for secret_type, pattern in regex_patterns:
                        match = pattern.search(body)
                        if match:
                            candidate = match.group(0)
                            
                            # Additional validation to ensure it's not a placeholder
                            if not any(placeholder in candidate.upper() for placeholder in [
                                "YOUR_", "REPLACE_", "EXAMPLE_", "TEST_", "PLACEHOLDER",
                                "DUMMY_", "FAKE_", "SAMPLE_", "CHANGE_", "INSERT_", "ADD_"
                            ]):
                                partial_results.append({
                                    "url": url,
                                    "title": issue["title"],
                                    "created_at": issue["created_at"],
                                    "repository_url": issue["repository_url"],
                                    "labels": ', '.join(label["name"] for label in issue["labels"]),
                                    "score": issue["score"],
                                    "secret_type": secret_type,
                                    "candidate_string": candidate,
                                    "body": body
                                })
                                seen_urls.add(url)
                                break

                # Periodic save
                if len(partial_results) >= SAVE_EVERY:
                    write_partial_results(partial_results, write_header=not header_written)
                    header_written = True
                    print(f"💾 Saved {len(partial_results)} matches to CSV.")
                    partial_results.clear()

                time.sleep(1)
            except Exception as e:
                print(f"❌ Error on page {page} for keyword '{keyword}': {e}")
                break

    # Final save
    if partial_results:
        write_partial_results(partial_results, write_header=not header_written)
        print(f"💾 Final flush: saved {len(partial_results)} remaining matches to CSV.")

    print(f"\n✅ Finished crawling. Total saved entries: {len(seen_urls)}")
    print("🎯 Results should contain fewer false positives due to stricter filtering")

if __name__ == "__main__":
    crawl()
