#!/usr/bin/env python3
"""
Script to fetch issues from GitHub repos and check them against regex patterns.
"""

import csv
import re
import os
import json
from typing import List, Set, Dict
import pandas as pd
from github import Github, GithubException
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN', '')  # GitHub token from .env file
ISSUES_PER_REPO = 20
REPOS_FILE = 'github_repo_urls.txt'
EXCLUDED_URLS_FILE = 'urls.txt'
REGEX_FILE = 'Secret-Regular-Expression.xlsx'
OUTPUT_FILE = 'issues_with_matches.csv'
CHECKPOINT_FILE = 'scan_checkpoint.json'
CHECKPOINT_INTERVAL = 1  # Save checkpoint every N repos


def load_repos(filepath: str) -> List[str]:
    """Load repository URLs from file."""
    with open(filepath, 'r') as f:
        repos = [line.strip() for line in f if line.strip()]
    return repos


def load_excluded_urls(filepath: str) -> Set[str]:
    """Load excluded issue URLs from file."""
    with open(filepath, 'r') as f:
        urls = {line.strip() for line in f if line.strip()}
    return urls


def load_checkpoint() -> Dict:
    """Load checkpoint data if exists."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
                print(f"✓ Loaded checkpoint: {checkpoint['completed_repos']} repos already processed")
                return checkpoint
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    return {'completed_repos': 0, 'processed_repo_urls': [], 'results': []}


def save_checkpoint(completed_repos: int, processed_repo_urls: List[str], results: List[Dict]):
    """Save checkpoint data."""
    try:
        checkpoint = {
            'completed_repos': completed_repos,
            'processed_repo_urls': processed_repo_urls,
            'results': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        print(f"  💾 Checkpoint saved: {completed_repos} repos processed")
    except Exception as e:
        print(f"  Warning: Could not save checkpoint: {e}")


def load_regex_patterns(filepath: str) -> List[Dict[str, str]]:
    """Load regex patterns from Excel file."""
    try:
        df = pd.read_excel(filepath)
        patterns = []
        
        # Use the specified column names
        regex_column = "Regular Expression"
        name_column = "Secret Type"
        
        # Verify columns exist
        if regex_column not in df.columns:
            print(f"Error: Column '{regex_column}' not found in Excel file.")
            print(f"Available columns: {df.columns.tolist()}")
            return []
        
        if name_column not in df.columns:
            print(f"Warning: Column '{name_column}' not found. Using index as name.")
            name_column = None
        
        for idx, row in df.iterrows():
            regex_pattern = str(row[regex_column])
            if pd.notna(regex_pattern) and regex_pattern.strip():
                pattern_dict = {
                    'pattern': regex_pattern.strip(),
                    'name': str(row[name_column]).strip() if name_column and pd.notna(row[name_column]) else f'Pattern_{idx+1}'
                }
                patterns.append(pattern_dict)
        
        print(f"Loaded {len(patterns)} regex patterns from {filepath}")
        return patterns
    except Exception as e:
        print(f"Error loading regex patterns: {e}")
        return []


def extract_repo_info(repo_url: str) -> tuple:
    """Extract owner and repo name from GitHub URL."""
    parts = repo_url.rstrip('/').split('/')
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return None, None


def check_text_against_patterns(text: str, patterns: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Check text against all regex patterns and return matches."""
    matches = []
    
    for pattern_dict in patterns:
        try:
            pattern = pattern_dict['pattern']
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                matches.append({
                    'secret_type': pattern_dict['name'],
                    'candidate_string': pattern
                })
        except re.error as e:
            # Skip invalid regex patterns
            continue
    
    return matches


def fetch_and_scan_issues(repos: List[str], excluded_urls: Set[str], patterns: List[Dict[str, str]], checkpoint: Dict = None):
    """Fetch issues from repos and scan for regex matches."""
    
    if not GITHUB_TOKEN:
        print("Warning: No GITHUB_TOKEN found. API rate limits will be very restrictive.")
        print("Set GITHUB_TOKEN environment variable to increase rate limits.")
        g = Github()
    else:
        g = Github(GITHUB_TOKEN)
    
    # Load checkpoint data
    if checkpoint is None:
        checkpoint = {'completed_repos': 0, 'processed_repo_urls': [], 'results': []}
    
    results = checkpoint.get('results', [])
    processed_repo_urls = set(checkpoint.get('processed_repo_urls', []))
    start_idx = checkpoint.get('completed_repos', 0)
    
    total_repos = len(repos)
    
    if start_idx > 0:
        print(f"\n🔄 Resuming from checkpoint: Starting at repo {start_idx + 1}/{total_repos}")
    
    for repo_idx, repo_url in enumerate(repos, 1):
        # Skip already processed repos
        if repo_url in processed_repo_urls:
            continue
        
        owner, repo_name = extract_repo_info(repo_url)
        
        if not owner or not repo_name:
            print(f"Skipping invalid URL: {repo_url}")
            continue
        
        print(f"\n[{repo_idx}/{total_repos}] Processing: {owner}/{repo_name}")
        
        try:
            repo = g.get_repo(f"{owner}/{repo_name}")
            issues = repo.get_issues(state='all', sort='created', direction='desc')
            
            issue_count = 0
            for issue in issues:
                if issue_count >= ISSUES_PER_REPO:
                    break
                
                # Skip pull requests - only process actual issues
                if issue.pull_request is not None:
                    continue
                
                issue_url = issue.html_url
                
                # Skip if URL is in excluded list
                if issue_url in excluded_urls:
                    continue
                
                # Get issue body (description)
                issue_body = issue.body or ""
                
                # Check against all regex patterns
                matches = check_text_against_patterns(issue_body, patterns)
                
                if matches:
                    # Limit to maximum 10 candidate strings per issue
                    matches_to_save = matches[:10]
                    
                    for match in matches_to_save:
                        results.append({
                            'issue_url': issue_url,
                            'issue_title': issue.title,
                            'issue_body': issue_body,
                            'secret_type': match['secret_type'],
                            'candidate_string': match['candidate_string'],
                            'repo': f"{owner}/{repo_name}"
                        })
                    
                    total_matches = len(matches)
                    saved_matches = len(matches_to_save)
                    if total_matches > 10:
                        print(f"  ✓ Issue #{issue.number}: {saved_matches}/5 pattern(s) saved ({total_matches} total found)")
                    else:
                        print(f"  ✓ Issue #{issue.number}: {saved_matches} pattern(s) matched")
                
                issue_count += 1
            
            print(f"  Scanned {issue_count} issues from {owner}/{repo_name}")
            
            # Mark repo as processed
            processed_repo_urls.add(repo_url)
            
            # Save checkpoint at intervals
            if repo_idx % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(repo_idx, list(processed_repo_urls), results)
            
            # Add small delay to avoid rate limiting
            time.sleep(0.5)
            
        except GithubException as e:
            print(f"  ✗ Error accessing {owner}/{repo_name}: {e}")
            if e.status == 403:
                print("  Rate limit exceeded. Waiting 60 seconds...")
                time.sleep(60)
            # Mark as processed even on error to avoid retrying
            processed_repo_urls.add(repo_url)
        except Exception as e:
            print(f"  ✗ Unexpected error for {owner}/{repo_name}: {e}")
            processed_repo_urls.add(repo_url)
    
    # Save final checkpoint
    save_checkpoint(len(processed_repo_urls), list(processed_repo_urls), results)
    
    return results


def save_results(results: List[Dict], output_file: str):
    """Save results to CSV file."""
    if not results:
        print("\nNo matches found.")
        return
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['repo', 'issue_url', 'issue_title', 'issue_body', 'secret_type', 'candidate_string']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Saved {len(results)} matches to {output_file}")


def main():
    """Main execution function."""
    print("=" * 70)
    print("GitHub Issues Secret Scanner (with Checkpoint Support)")
    print("=" * 70)
    
    # Load data
    print("\nLoading configuration...")
    repos = load_repos(REPOS_FILE)
    excluded_urls = load_excluded_urls(EXCLUDED_URLS_FILE)
    patterns = load_regex_patterns(REGEX_FILE)
    
    if not patterns:
        print("Error: No regex patterns loaded. Please check your Excel file.")
        return
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    
    print(f"\nConfiguration:")
    print(f"  - Repositories: {len(repos)}")
    print(f"  - Excluded URLs: {len(excluded_urls)}")
    print(f"  - Regex patterns: {len(patterns)}")
    print(f"  - Issues per repo: {ISSUES_PER_REPO}")
    print(f"  - Checkpoint interval: Every {CHECKPOINT_INTERVAL} repos")
    
    # Fetch and scan issues
    print("\nStarting scan...")
    results = fetch_and_scan_issues(repos, excluded_urls, patterns, checkpoint)
    
    # Save results
    save_results(results, OUTPUT_FILE)
    
    # Clean up checkpoint file after successful completion
    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
            print("✓ Checkpoint file removed (scan completed successfully)")
        except Exception as e:
            print(f"Warning: Could not remove checkpoint file: {e}")
    
    print("\n" + "=" * 70)
    print("Scan complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
