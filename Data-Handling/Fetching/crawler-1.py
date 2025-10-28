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
OUTPUT_CSV = "filtered_github_secrets.csv"
REGEX_FILE = "Secret-Regular-Expression.xlsx"
SAVE_EVERY = 1000  
SKIP_URLS_FILE = "alltp.txt"
SEARCH_KEYWORDS = [
    # === Core secret terms with all combinations ===
     "secret=", "secret =", "secret:", "secret :", "secret ==", "secret: ", "secret = ",
    '"secret"', "'secret'", '"secret":', "'secret':", '"secret"=', "'secret'=",
     "SECRET=", "SECRET =", "SECRET:", "SECRET :", "SECRET: ", "SECRET = ",
    '"SECRET"', "'SECRET'", '"SECRET":', "'SECRET':", '"SECRET"=', "'SECRET'=",
    
    # === API Key variations ===
     "api_key=", "api_key =", "api_key:", "api_key :", "api_key: ", "api_key = ",
     "apikey=", "apikey =", "apikey:", "apikey :", "apikey: ", "apikey = ",
     "apiKey=", "apiKey =", "apiKey:", "apiKey :", "apiKey: ", "apiKey = ",
     "API_KEY=", "API_KEY =", "API_KEY:", "API_KEY :", "API_KEY: ", "API_KEY = ",
    "APIKEY=", "APIKEY =", "APIKEY:", "APIKEY :", "APIKEY: ", "APIKEY = ",
     "'api_key'", '"api_key":', "'api_key':", '"api_key"=', "'api_key'=",
     "'apikey'", '"apikey":', "'apikey':", '"apikey"=', "'apikey'=",
    "'API_KEY'", '"API_KEY":', "'API_KEY':", '"API_KEY"=', "'API_KEY'=",
    
    # === Token variations ===
     "token=", "token =", "token:", "token :", "token: ", "token = ",
 "Token=", "Token =", "Token:", "Token :", "Token: ", "Token = ",
    "TOKEN=", "TOKEN =", "TOKEN:", "TOKEN :", "TOKEN: ", "TOKEN = ",
     '"token":', "'token':", '"token"=', "'token'=",
     '"Token":', "'Token':", '"Token"=', "'Token'=",
     '"TOKEN":', "'TOKEN':", '"TOKEN"=', "'TOKEN'=",
    
    # === Key variations ===
     "key=", "key =", "key:", "key :", "key: ", "key = ",
     "Key=", "Key =", "Key:", "Key :", "Key: ", "Key = ",
     "KEY=", "KEY =", "KEY:", "KEY :", "KEY: ", "KEY = ",
     "'key'", '"key":', "'key':", '"key"=', "'key'=",
    "'Key'", '"Key":', "'Key':", '"Key"=', "'Key'=",
    '"KEY":', "'KEY':", '"KEY"=', "'KEY'=",
    
    # === Password variations ===
     "password=", "password =", "password:", "password :", "password: ", "password = ",
     "Password=", "Password =", "Password:", "Password :", "Password: ", "Password = ",
     "PASSWORD=", "PASSWORD =", "PASSWORD:", "PASSWORD :", "PASSWORD: ", "PASSWORD = ",
 "passwd=", "passwd =", "passwd:", "passwd :", "passwd: ", "passwd = ",
     "pwd=", "pwd =", "pwd:", "pwd :", "pwd: ", "pwd = ",
   '"password":', "'password':", '"password"=', "'password'=",
     '"passwd":', "'passwd':", '"passwd"=', "'passwd'=",
    '"pwd":', "'pwd':", '"pwd"=', "'pwd'=",
    
    # === Auth variations ===
    "auth=", "auth =", "auth:", "auth :", "auth: ", "auth = ",
    "Auth=", "Auth =", "Auth:", "Auth :", "Auth: ", "Auth = ",
    "AUTH=", "AUTH =", "AUTH:", "AUTH :", "AUTH: ", "AUTH = ",
    "authorization=", "Authorization=", "AUTHORIZATION=",
    "authorization:", "Authorization:", "AUTHORIZATION:",
     '"auth":', "'auth':", '"auth"=', "'auth'=",
    '"authorization":', "'authorization':",
    
    # === Bearer variations ===
    "bearer=", "bearer =", "bearer:", "bearer :", "bearer: ", "bearer = ",
    "Bearer=", "Bearer =", "Bearer:", "Bearer :", "Bearer: ", "Bearer = ",
    "BEARER=", "BEARER =", "BEARER:", "BEARER :", "BEARER: ", "BEARER = ",
    
    # === Client credentials ===
    "client_secret=", "client_secret =", "client_secret:", "client_secret :",
    "clientSecret=", "clientSecret =", "clientSecret:", "clientSecret :",
    "CLIENT_SECRET=", "CLIENT_SECRET =", "CLIENT_SECRET:", "CLIENT_SECRET :",
    "client_id=", "client_id =", "client_id:", "client_id :",
    "clientId=", "clientId =", "clientId:", "clientId :",
    "CLIENT_ID=", "CLIENT_ID =", "CLIENT_ID:", "CLIENT_ID :",
     '"client_secret":', "'client_secret':",
    
    # === Access token variations ===
     "access_token=", "access_token =", "access_token:", "access_token :",
    "accessToken=", "accessToken =", "accessToken:", "accessToken :",
     "ACCESS_TOKEN=", "ACCESS_TOKEN =", "ACCESS_TOKEN:", "ACCESS_TOKEN :",
    "refresh_token=", "refresh_token =", "refresh_token:", "refresh_token :",
     "refreshToken=", "refreshToken =", "refreshToken:", "refreshToken :",
    "REFRESH_TOKEN=", "REFRESH_TOKEN =", "REFRESH_TOKEN:", "REFRESH_TOKEN :",
    
    # === Platform-specific keys with combinations ===
    
    
     "aws_secret_access_key=", "aws_secret_access_key =", "aws_secret_access_key:",
     "AWS_SECRET_ACCESS_KEY=", "AWS_SECRET_ACCESS_KEY =", "AWS_SECRET_ACCESS_KEY:",
    "github_token=", "github_token =", "github_token:", "GITHUB_TOKEN",
     "slack_token=", "slack_token =", "slack_token:", "SLACK_TOKEN",
    "discord_token=", "discord_token =", "discord_token:", "DISCORD_TOKEN",
     "google_api_key=", "google_api_key =", "google_api_key:", "GOOGLE_API_KEY",
     "openai_api_key=", "openai_api_key =", "openai_api_key:", "OPENAI_API_KEY",
    "stripe_key=", "stripe_key =", "stripe_key:", "STRIPE_KEY",
    "twilio_auth_token", "twilio_auth_token=", "twilio_auth_token =", "twilio_auth_token:",
    "sendgrid_api_key", "sendgrid_api_key=", "sendgrid_api_key =", "sendgrid_api_key:",
      # Azure
   "azure_storage_key=", "azure_storage_key =", "azure_storage_key:",
     "AZURE_STORAGE_KEY=", "AZURE_STORAGE_KEY =", "AZURE_STORAGE_KEY:",
    "azure_client_secret=", "azure_client_secret =", "azure_client_secret:",
    "AZURE_CLIENT_SECRET=", "AZURE_CLIENT_SECRET =", "AZURE_CLIENT_SECRET:",

    # Firebase
    "firebase_api_key=", "firebase_api_key =", "firebase_api_key:",
    "FIREBASE_API_KEY=", "FIREBASE_API_KEY =", "FIREBASE_API_KEY:",

    # Heroku
     "heroku_api_key=", "heroku_api_key =", "heroku_api_key:",
     "HEROKU_API_KEY=", "HEROKU_API_KEY =", "HEROKU_API_KEY:",

    # DigitalOcean
    "do_api_token=", "do_api_token =", "do_api_token:",
    "DO_API_TOKEN=", "DO_API_TOKEN =", "DO_API_TOKEN:",

    # MongoDB Atlas
    "mongodb_uri=", "mongodb_uri =", "mongodb_uri:",
    "MONGODB_URI=", "MONGODB_URI =", "MONGODB_URI:",

    # General JWTs & Tokens
    "jwt_secret=", "jwt_secret =", "jwt_secret:",
    "JWT_SECRET=", "JWT_SECRET =", "JWT_SECRET:",
    "api_token=", "api_token =", "api_token:",
    "API_TOKEN=", "API_TOKEN =", "API_TOKEN:",
    
    "gcp_private_key", "GCP_PRIVATE_KEY",
    "google_api_key", "GOOGLE_API_KEY",
    "google_oauth_client_id", "GOOGLE_OAUTH_CLIENT_ID",
    "google_oauth_client_secret", "GOOGLE_OAUTH_CLIENT_SECRET",
    
    # === Telegram ===
    "telegram_token", "TELEGRAM_TOKEN",
    "telegram_bot_token", "TELEGRAM_BOT_TOKEN",
    
    # === Database ===
    "mongodb_uri", "MONGODB_URI",
    "mysql_url", "MYSQL_URL",
    "postgres_url", "POSTGRES_URL",
    "redis_url", "REDIS_URL",

    # === Docker / Kubernetes ===
    "dockerhub_username", "DOCKERHUB_USERNAME",
    "dockerhub_password", "DOCKERHUB_PASSWORD",
    "k8s_token", "K8S_TOKEN",
    "kubernetes_token", "KUBERNETES_TOKEN",

    # === DigitalOcean / Linode / Vultr ===
    "do_api_token", "DO_API_TOKEN",
    "linode_api_token", "LINODE_API_TOKEN",
    "vultr_api_key", "VULTR_API_KEY",

    # === Heroku ===
    "heroku_api_key", "HEROKU_API_KEY",

    # === Cloudflare ===
    "cloudflare_api_token", "CLOUDFLARE_API_TOKEN",
    "cloudflare_api_key", "CLOUDFLARE_API_KEY",
    "cloudflare_email", "CLOUDFLARE_EMAIL",
    
    # === Private key variations ===
    "private_key", "private_key=", "private_key =", "private_key:", "private_key :",
    "privateKey", "privateKey=", "privateKey =", "privateKey:", "privateKey :",
    "PRIVATE_KEY", "PRIVATE_KEY=", "PRIVATE_KEY =", "PRIVATE_KEY:", "PRIVATE_KEY :",
    "ssh_private_key", "rsa_private_key", "dsa_private_key",
    '"private_key"', "'private_key'", '"private_key":', "'private_key':",
    
    # === Database credentials ===
    "db_password", "db_password=", "db_password =", "db_password:", "db_password :",
    "db_user", "db_user=", "db_user =", "db_user:", "db_user :",
    "database_url", "database_url=", "database_url =", "database_url:",
    "connection_string", "connection_string=", "connection_string =", "connection_string:",
    "mysql_password", "postgres_password", "mongodb_uri", "redis_password",
    
    # === Environment variable patterns ===
    "export SECRET", "export API_KEY", "export TOKEN", "export PASSWORD",
    "export GITHUB_TOKEN", "export OPENAI_API_KEY", "export AWS_ACCESS_KEY_ID",
    "process.env.SECRET", "process.env.API_KEY", "process.env.TOKEN",
    "os.environ", "getenv", "$SECRET", "$API_KEY", "$TOKEN",
    
    # === Common secret prefixes/patterns ===
    "AKIA", "ghp_", "ghs_", "github_pat_", "xoxb-", "xoxp-", "sk-", "pk_",
    "ya29.", "AIza", "sk_live_", "sk_test_", "pk_live_", "pk_test_",
    
    # === Header patterns ===
    "X-API-Key", "X-Auth-Token", "X-Access-Token", "X-Secret-Key",
    "Authorization:", "Authorization =", "Bearer ", "Basic ",
    
    # === Config file indicators ===
    ".env", ".env.local", ".env.production", ".env.development",
    "config.json", "config.yaml", "config.yml", "secrets.json",
    "credentials.json", ".aws/credentials", "firebase-config.json",
    
    # === Common secret values/patterns ===
    "BEGIN RSA PRIVATE KEY", "BEGIN PRIVATE KEY", "BEGIN CERTIFICATE",
    "-----BEGIN", "-----END", "mysql://", "postgresql://", "mongodb://",
    
    # === Webhook and callback secrets ===
    "webhook_secret", "webhook_token", "callback_secret", "signing_secret",
    
    # === Error patterns that might expose secrets ===
    "invalid api key", "invalid token", "expired token", "authentication failed",
    "unauthorized", "access denied", "forbidden", "token expired",
    "credentials not found", "invalid credentials", "login failed",
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

    print(f"🔍 Searching using {len(SEARCH_KEYWORDS)} keywords and applying {len(regex_patterns)} regex patterns...")

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
                    for secret_type, pattern in regex_patterns:
                        match = pattern.search(body)
                        if match:
                            partial_results.append({
                                "url": url,
                                "title": issue["title"],
                                "created_at": issue["created_at"],
                                "repository_url": issue["repository_url"],
                                "labels": ', '.join(label["name"] for label in issue["labels"]),
                                "score": issue["score"],
                                "secret_type": secret_type,
                                "candidate_string": match.group(0),
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

if __name__ == "__main__":
    crawl()
