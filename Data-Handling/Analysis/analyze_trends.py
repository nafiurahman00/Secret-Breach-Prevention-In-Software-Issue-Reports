#!/usr/bin/env python3
"""
GitHub Secrets Leak Trend Analysis

This script analyzes the crawled metadata to identify trends and patterns
in secret leaks across GitHub repositories.
"""

import json
import os
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import pandas as pd


class TrendAnalyzer:
    """Analyzer for GitHub secret leak trends"""
    
    def __init__(self, metadata_dir: str = 'metadata', output_dir: str = 'analysis'):
        self.metadata_dir = metadata_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.repos_data = []
        self.issues_data = []
    
    def load_metadata(self):
        """Load metadata from JSON files"""
        print("📂 Loading metadata...")
        
        repos_file = os.path.join(self.metadata_dir, 'repos_metadata.json')
        issues_file = os.path.join(self.metadata_dir, 'issues_metadata.json')
        
        if os.path.exists(repos_file):
            with open(repos_file, 'r') as f:
                self.repos_data = json.load(f)
            print(f"✅ Loaded {len(self.repos_data)} repositories")
        else:
            print("⚠️  repos_metadata.json not found")
        
        if os.path.exists(issues_file):
            with open(issues_file, 'r') as f:
                self.issues_data = json.load(f)
            print(f"✅ Loaded {len(self.issues_data)} issues")
        else:
            print("⚠️  issues_metadata.json not found")
    
    def classify_issue_type(self, issue: Dict) -> str:
        """Classify issue as bug, feature, security, documentation, etc."""
        labels = [label.lower() for label in issue.get('labels', [])]
        title = issue.get('title', '').lower()
        
        # Security-related
        security_keywords = ['security', 'vulnerability', 'cve', 'leak', 'exposed', 'credential', 'secret', 'token', 'key', 'password']
        if any(kw in ' '.join(labels) for kw in security_keywords) or any(kw in title for kw in security_keywords):
            return 'security'
        
        # Bug-related
        bug_keywords = ['bug', 'error', 'issue', 'problem', 'fix', 'broken']
        if 'bug' in labels or any(kw in title for kw in bug_keywords):
            return 'bug'
        
        # Feature-related
        feature_keywords = ['feature', 'enhancement', 'improvement', 'request', 'add', 'new']
        if any(kw in labels for kw in ['feature', 'enhancement']) or any(kw in title for kw in feature_keywords):
            return 'feature'
        
        # Documentation
        doc_keywords = ['documentation', 'docs', 'readme']
        if any(kw in labels for kw in doc_keywords) or any(kw in title for kw in doc_keywords):
            return 'documentation'
        
        # Question/Support
        if 'question' in labels or 'help' in labels or '?' in title:
            return 'question'
        
        return 'other'
    
    def analyze_issue_types(self) -> Dict:
        """Analyze distribution of issue types"""
        print("\n📊 Analyzing issue types...")
        
        issue_types = Counter()
        for issue in self.issues_data:
            issue_type = self.classify_issue_type(issue)
            issue_types[issue_type] += 1
        
        total = sum(issue_types.values())
        result = {
            'counts': dict(issue_types),
            'percentages': {k: (v/total)*100 for k, v in issue_types.items()},
            'total': total
        }
        
        print(f"\n📈 Issue Type Distribution:")
        for issue_type, count in issue_types.most_common():
            pct = (count/total)*100
            print(f"   {issue_type.capitalize():<15}: {count:>5} ({pct:>5.1f}%)")
        
        return result
    
    def analyze_leak_by_repo_size(self) -> Dict:
        """Analyze leak frequency by repository size"""
        print("\n📊 Analyzing leaks by repository size...")
        
        # Create a mapping of repo to its issues
        repo_issues = defaultdict(list)
        for issue in self.issues_data:
            repo_full_name = issue.get('repo_full_name')
            if repo_full_name:
                repo_issues[repo_full_name].append(issue)
        
        # Map repos to size categories
        size_buckets = {
            'tiny': (0, 100),           # < 100 KB
            'small': (100, 1000),       # 100 KB - 1 MB
            'medium': (1000, 10000),    # 1 MB - 10 MB
            'large': (10000, 100000),   # 10 MB - 100 MB
            'huge': (100000, float('inf'))  # > 100 MB
        }
        
        leaks_by_size = defaultdict(list)
        repos_by_size = defaultdict(int)
        
        for repo in self.repos_data:
            size = repo.get('size', 0)
            full_name = repo.get('full_name')
            
            for bucket, (min_size, max_size) in size_buckets.items():
                if min_size <= size < max_size:
                    repos_by_size[bucket] += 1
                    leak_count = len(repo_issues.get(full_name, []))
                    if leak_count > 0:
                        leaks_by_size[bucket].append(leak_count)
                    break
        
        result = {}
        for bucket in size_buckets.keys():
            leak_counts = leaks_by_size[bucket]
            repo_count = repos_by_size[bucket]
            
            if leak_counts:
                avg_leaks = sum(leak_counts) / len(leak_counts)
                total_leaks = sum(leak_counts)
                max_leaks = max(leak_counts)
            else:
                avg_leaks = total_leaks = max_leaks = 0
            
            result[bucket] = {
                'repo_count': repo_count,
                'repos_with_leaks': len(leak_counts),
                'total_leaks': total_leaks,
                'avg_leaks_per_repo': avg_leaks,
                'max_leaks': max_leaks
            }
        
        print(f"\n📈 Leaks by Repository Size:")
        for bucket, stats in result.items():
            print(f"\n   {bucket.upper()}:")
            print(f"      Repos: {stats['repo_count']}")
            print(f"      Repos with leaks: {stats['repos_with_leaks']}")
            print(f"      Total leaks: {stats['total_leaks']}")
            print(f"      Avg leaks/repo: {stats['avg_leaks_per_repo']:.2f}")
        
        return result
    
    def analyze_leak_by_stars(self) -> Dict:
        """Analyze leak frequency by repository popularity (stars)"""
        print("\n📊 Analyzing leaks by repository stars...")
        
        # Create a mapping of repo to its issues
        repo_issues = defaultdict(list)
        for issue in self.issues_data:
            repo_full_name = issue.get('repo_full_name')
            if repo_full_name:
                repo_issues[repo_full_name].append(issue)
        
        # Star categories
        star_buckets = {
            'zero': (0, 1),         # 0 stars
            'minimal': (1, 10),        # 1-9 stars
            'low': (10, 100),          # 10-99 stars
            'medium': (100, 1000),     # 100-999 stars
            'high': (1000, 10000),     # 1k-10k stars
            'very_high': (10000, float('inf'))  # 10k+ stars
        }
        
        leaks_by_stars = defaultdict(list)
        repos_by_stars = defaultdict(int)
        
        for repo in self.repos_data:
            stars = repo.get('stars', 0)
            full_name = repo.get('full_name')
            
            for bucket, (min_stars, max_stars) in star_buckets.items():
                if min_stars <= stars < max_stars:
                    repos_by_stars[bucket] += 1
                    leak_count = len(repo_issues.get(full_name, []))
                    if leak_count > 0:
                        leaks_by_stars[bucket].append(leak_count)
                    break
        
        result = {}
        for bucket in star_buckets.keys():
            leak_counts = leaks_by_stars[bucket]
            repo_count = repos_by_stars[bucket]
            
            if leak_counts:
                avg_leaks = sum(leak_counts) / len(leak_counts)
                total_leaks = sum(leak_counts)
                max_leaks = max(leak_counts)
            else:
                avg_leaks = total_leaks = max_leaks = 0
            
            result[bucket] = {
                'repo_count': repo_count,
                'repos_with_leaks': len(leak_counts),
                'total_leaks': total_leaks,
                'avg_leaks_per_repo': avg_leaks,
                'max_leaks': max_leaks
            }
        
        print(f"\n📈 Leaks by Repository Stars:")
        for bucket, stats in result.items():
            print(f"\n   {bucket.upper().replace('_', ' ')}:")
            print(f"      Repos: {stats['repo_count']}")
            print(f"      Repos with leaks: {stats['repos_with_leaks']}")
            print(f"      Total leaks: {stats['total_leaks']}")
            print(f"      Avg leaks/repo: {stats['avg_leaks_per_repo']:.2f}")
        
        return result
    
    def analyze_leak_over_time(self) -> Dict:
        """Analyze leak occurrence over time"""
        print("\n📊 Analyzing leaks over time...")
        
        # Group issues by year and month
        leaks_by_year = defaultdict(int)
        leaks_by_month = defaultdict(int)
        leaks_by_quarter = defaultdict(int)
        
        for issue in self.issues_data:
            created_at = issue.get('created_at')
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    year = dt.year
                    month = dt.strftime('%Y-%m')
                    quarter = f"{year}-Q{(dt.month-1)//3 + 1}"
                    
                    leaks_by_year[year] += 1
                    leaks_by_month[month] += 1
                    leaks_by_quarter[quarter] += 1
                except Exception as e:
                    pass
        
        result = {
            'by_year': dict(sorted(leaks_by_year.items())),
            'by_month': dict(sorted(leaks_by_month.items())),
            'by_quarter': dict(sorted(leaks_by_quarter.items()))
        }
        
        print(f"\n📈 Leaks by Year:")
        for year, count in sorted(leaks_by_year.items()):
            print(f"   {year}: {count}")
        
        print(f"\n📈 Recent Quarters:")
        for quarter, count in sorted(leaks_by_quarter.items())[-8:]:
            print(f"   {quarter}: {count}")
        
        return result
    
    def analyze_top_languages(self) -> Dict:
        """Analyze leaks by programming language"""
        print("\n📊 Analyzing leaks by programming language...")
        
        # Create a mapping of repo to its issues
        repo_issues = defaultdict(list)
        for issue in self.issues_data:
            repo_full_name = issue.get('repo_full_name')
            if repo_full_name:
                repo_issues[repo_full_name].append(issue)
        
        leaks_by_language = defaultdict(int)
        repos_by_language = defaultdict(int)
        
        for repo in self.repos_data:
            language = repo.get('language')
            if not language:
                language = 'Unknown'
            
            full_name = repo.get('full_name')
            repos_by_language[language] += 1
            
            leak_count = len(repo_issues.get(full_name, []))
            leaks_by_language[language] += leak_count
        
        result = {
            'leaks': dict(sorted(leaks_by_language.items(), key=lambda x: x[1], reverse=True)[:15]),
            'repos': dict(sorted(repos_by_language.items(), key=lambda x: x[1], reverse=True)[:15])
        }
        
        print(f"\n📈 Top Languages with Leaks:")
        for lang, count in sorted(leaks_by_language.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {lang:<20}: {count} leaks")
        
        return result
    
    def analyze_repo_characteristics(self) -> Dict:
        """Analyze characteristics of repos with leaks"""
        print("\n📊 Analyzing repository characteristics...")
        
        # Create a mapping of repo to its issues
        repo_issues = defaultdict(list)
        for issue in self.issues_data:
            repo_full_name = issue.get('repo_full_name')
            if repo_full_name:
                repo_issues[repo_full_name].append(issue)
        
        repos_with_leaks = []
        repos_without_leaks = []
        
        for repo in self.repos_data:
            full_name = repo.get('full_name')
            if repo_issues.get(full_name):
                repos_with_leaks.append(repo)
            else:
                repos_without_leaks.append(repo)
        
        def calculate_stats(repos):
            if not repos:
                return {}
            
            return {
                'count': len(repos),
                'avg_stars': sum(r.get('stars', 0) for r in repos) / len(repos),
                'avg_size': sum(r.get('size', 0) for r in repos) / len(repos),
                'avg_forks': sum(r.get('forks', 0) for r in repos) / len(repos),
                'is_fork_pct': sum(1 for r in repos if r.get('is_fork', False)) / len(repos) * 100,
                'is_archived_pct': sum(1 for r in repos if r.get('is_archived', False)) / len(repos) * 100,
            }
        
        result = {
            'with_leaks': calculate_stats(repos_with_leaks),
            'without_leaks': calculate_stats(repos_without_leaks)
        }
        
        print(f"\n📈 Repository Characteristics:")
        print(f"\n   Repos WITH leaks ({result['with_leaks']['count']}):")
        print(f"      Avg stars: {result['with_leaks']['avg_stars']:.1f}")
        print(f"      Avg size (KB): {result['with_leaks']['avg_size']:.1f}")
        print(f"      Avg forks: {result['with_leaks']['avg_forks']:.1f}")
        print(f"      Fork %: {result['with_leaks']['is_fork_pct']:.1f}%")
        print(f"      Archived %: {result['with_leaks']['is_archived_pct']:.1f}%")
        
        if result['without_leaks']['count'] > 0:
            print(f"\n   Repos WITHOUT leaks ({result['without_leaks']['count']}):")
            print(f"      Avg stars: {result['without_leaks']['avg_stars']:.1f}")
            print(f"      Avg size (KB): {result['without_leaks']['avg_size']:.1f}")
            print(f"      Avg forks: {result['without_leaks']['avg_forks']:.1f}")
            print(f"      Fork %: {result['without_leaks']['is_fork_pct']:.1f}%")
            print(f"      Archived %: {result['without_leaks']['is_archived_pct']:.1f}%")
        
        return result
    
    def analyze_issue_response_time(self) -> Dict:
        """Analyze how quickly issues are responded to and closed"""
        print("\n📊 Analyzing issue response times...")
        
        response_times = []
        close_times = []
        
        for issue in self.issues_data:
            created_at = issue.get('created_at')
            closed_at = issue.get('closed_at')
            updated_at = issue.get('updated_at')
            
            if created_at and updated_at:
                try:
                    created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    updated = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                    response_hours = (updated - created).total_seconds() / 3600
                    if response_hours > 0:
                        response_times.append(response_hours)
                except:
                    pass
            
            if created_at and closed_at:
                try:
                    created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    closed = datetime.fromisoformat(closed_at.replace('Z', '+00:00'))
                    close_hours = (closed - created).total_seconds() / 3600
                    if close_hours > 0:
                        close_times.append(close_hours)
                except:
                    pass
        
        result = {}
        
        if response_times:
            result['response'] = {
                'avg_hours': sum(response_times) / len(response_times),
                'median_hours': sorted(response_times)[len(response_times)//2],
                'min_hours': min(response_times),
                'max_hours': max(response_times)
            }
            
            print(f"\n📈 Response Times:")
            print(f"   Average: {result['response']['avg_hours']:.1f} hours")
            print(f"   Median: {result['response']['median_hours']:.1f} hours")
        
        if close_times:
            result['close'] = {
                'avg_hours': sum(close_times) / len(close_times),
                'median_hours': sorted(close_times)[len(close_times)//2],
                'min_hours': min(close_times),
                'max_hours': max(close_times),
                'closed_count': len(close_times),
                'closed_pct': len(close_times) / len(self.issues_data) * 100
            }
            
            print(f"\n📈 Close Times:")
            print(f"   Average: {result['close']['avg_hours']:.1f} hours ({result['close']['avg_hours']/24:.1f} days)")
            print(f"   Median: {result['close']['median_hours']:.1f} hours ({result['close']['median_hours']/24:.1f} days)")
            print(f"   Closed: {result['close']['closed_count']}/{len(self.issues_data)} ({result['close']['closed_pct']:.1f}%)")
        
        return result
    
    def run_full_analysis(self) -> Dict:
        """Run all analysis functions"""
        print("\n" + "=" * 80)
        print("📊 GITHUB SECRETS LEAK TREND ANALYSIS")
        print("=" * 80)
        
        self.load_metadata()
        
        results = {
            'summary': {
                'total_repos': len(self.repos_data),
                'total_issues': len(self.issues_data),
                'timestamp': datetime.now().isoformat()
            },
            'issue_types': self.analyze_issue_types(),
            'leak_by_size': self.analyze_leak_by_repo_size(),
            'leak_by_stars': self.analyze_leak_by_stars(),
            'leak_over_time': self.analyze_leak_over_time(),
            'top_languages': self.analyze_top_languages(),
            'repo_characteristics': self.analyze_repo_characteristics(),
            'response_times': self.analyze_issue_response_time()
        }
        
        # Save results
        output_file = os.path.join(self.output_dir, 'trend_analysis.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "=" * 80)
        print(f"✅ Analysis complete! Results saved to: {output_file}")
        print("=" * 80)
        
        return results


def main():
    analyzer = TrendAnalyzer()
    results = analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
