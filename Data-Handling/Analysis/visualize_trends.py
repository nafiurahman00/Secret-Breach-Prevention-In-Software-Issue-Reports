#!/usr/bin/env python3
"""
GitHub Secrets Leak Visualization

This script creates visualizations from the trend analysis results.
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict
import numpy as np


class LeakVisualizer:
    """Create visualizations for leak trend analysis"""
    
    def __init__(self, analysis_dir: str = 'analysis', output_dir: str = 'visualizations'):
        self.analysis_dir = analysis_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        self.data = None
    
    def load_analysis_data(self):
        """Load analysis results"""
        print("📂 Loading analysis data...")
        
        analysis_file = os.path.join(self.analysis_dir, 'trend_analysis.json')
        if not os.path.exists(analysis_file):
            raise FileNotFoundError(f"Analysis file not found: {analysis_file}")
        
        with open(analysis_file, 'r') as f:
            self.data = json.load(f)
        
        print(f"✅ Loaded analysis data")
    
    def plot_issue_types(self):
        """Plot issue type distribution"""
        print("📊 Creating issue type distribution chart...")
        
        issue_types = self.data['issue_types']['counts']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Pie chart
        colors = sns.color_palette("husl", len(issue_types))
        wedges, texts, autotexts = ax1.pie(
            issue_types.values(),
            labels=issue_types.keys(),
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        ax1.set_title('Issue Type Distribution', fontsize=14, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # Bar chart
        sorted_types = sorted(issue_types.items(), key=lambda x: x[1], reverse=True)
        types, counts = zip(*sorted_types)
        bars = ax2.bar(types, counts, color=colors)
        ax2.set_xlabel('Issue Type', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Issue Type Counts', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, '01_issue_types.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_path}")
    
    def plot_leak_by_repo_size(self):
        """Plot leak frequency by repository size"""
        print("📊 Creating leak by repo size chart...")
        
        size_data = self.data['leak_by_size']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        sizes = list(size_data.keys())
        
        # Total leaks by size
        total_leaks = [size_data[s]['total_leaks'] for s in sizes]
        axes[0, 0].bar(sizes, total_leaks, color=sns.color_palette("viridis", len(sizes)))
        axes[0, 0].set_title('Total Leaks by Repository Size', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Total Leaks', fontsize=11)
        axes[0, 0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(total_leaks):
            axes[0, 0].text(i, v, str(v), ha='center', va='bottom')
        
        # Average leaks per repo
        avg_leaks = [size_data[s]['avg_leaks_per_repo'] for s in sizes]
        axes[0, 1].bar(sizes, avg_leaks, color=sns.color_palette("mako", len(sizes)))
        axes[0, 1].set_title('Average Leaks per Repository', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Avg Leaks', fontsize=11)
        axes[0, 1].tick_params(axis='x', rotation=45)
        for i, v in enumerate(avg_leaks):
            axes[0, 1].text(i, v, f'{v:.2f}', ha='center', va='bottom')
        
        # Repos with leaks
        repos_with = [size_data[s]['repos_with_leaks'] for s in sizes]
        total_repos = [size_data[s]['repo_count'] for s in sizes]
        axes[1, 0].bar(sizes, repos_with, label='With Leaks', alpha=0.8)
        axes[1, 0].bar(sizes, total_repos, label='Total Repos', alpha=0.5)
        axes[1, 0].set_title('Repositories by Size', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Count', fontsize=11)
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend()
        
        # Leak percentage by size
        leak_pct = [(repos_with[i]/total_repos[i]*100) if total_repos[i] > 0 else 0 
                    for i in range(len(sizes))]
        bars = axes[1, 1].bar(sizes, leak_pct, color=sns.color_palette("rocket", len(sizes)))
        axes[1, 1].set_title('Percentage of Repos with Leaks', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Percentage (%)', fontsize=11)
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_ylim(0, 100)
        for i, v in enumerate(leak_pct):
            axes[1, 1].text(i, v, f'{v:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, '02_leak_by_size.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_path}")
    
    def plot_leak_by_stars(self):
        """Plot leak frequency by repository stars"""
        print("📊 Creating leak by repo stars chart...")
        
        stars_data = self.data['leak_by_stars']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        star_categories = list(stars_data.keys())
        
        # Total leaks by stars
        total_leaks = [stars_data[s]['total_leaks'] for s in star_categories]
        axes[0, 0].bar(star_categories, total_leaks, color=sns.color_palette("coolwarm", len(star_categories)))
        axes[0, 0].set_title('Total Leaks by Repository Popularity (Stars)', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Total Leaks', fontsize=11)
        axes[0, 0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(total_leaks):
            axes[0, 0].text(i, v, str(v), ha='center', va='bottom')
        
        # Average leaks per repo
        avg_leaks = [stars_data[s]['avg_leaks_per_repo'] for s in star_categories]
        axes[0, 1].bar(star_categories, avg_leaks, color=sns.color_palette("crest", len(star_categories)))
        axes[0, 1].set_title('Average Leaks per Repository', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Avg Leaks', fontsize=11)
        axes[0, 1].tick_params(axis='x', rotation=45)
        for i, v in enumerate(avg_leaks):
            axes[0, 1].text(i, v, f'{v:.2f}', ha='center', va='bottom')
        
        # Distribution of repos
        repos_with = [stars_data[s]['repos_with_leaks'] for s in star_categories]
        total_repos = [stars_data[s]['repo_count'] for s in star_categories]
        x = np.arange(len(star_categories))
        width = 0.35
        axes[1, 0].bar(x - width/2, repos_with, width, label='With Leaks', alpha=0.8)
        axes[1, 0].bar(x + width/2, total_repos, width, label='Total Repos', alpha=0.5)
        axes[1, 0].set_title('Repository Distribution by Stars', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Count', fontsize=11)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(star_categories, rotation=45)
        axes[1, 0].legend()
        
        # Leak rate by popularity
        leak_pct = [(repos_with[i]/total_repos[i]*100) if total_repos[i] > 0 else 0 
                    for i in range(len(star_categories))]
        axes[1, 1].plot(star_categories, leak_pct, marker='o', linewidth=2, markersize=8)
        axes[1, 1].fill_between(range(len(star_categories)), leak_pct, alpha=0.3)
        axes[1, 1].set_title('Leak Rate by Repository Popularity', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Leak Rate (%)', fontsize=11)
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_ylim(0, max(leak_pct) * 1.1 if leak_pct else 100)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, '03_leak_by_stars.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_path}")
    
    def plot_leak_over_time(self):
        """Plot leak occurrence over time"""
        print("📊 Creating leak over time chart...")
        
        time_data = self.data['leak_over_time']
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Yearly trend
        years = sorted([int(y) for y in time_data['by_year'].keys()])
        yearly_counts = [time_data['by_year'][str(y)] for y in years]
        
        axes[0].plot(years, yearly_counts, marker='o', linewidth=2, markersize=8, color='#2E86AB')
        axes[0].fill_between(years, yearly_counts, alpha=0.3, color='#2E86AB')
        axes[0].set_title('Secret Leaks Over Time (Yearly)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_ylabel('Number of Leaks', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels
        for x, y in zip(years, yearly_counts):
            axes[0].text(x, y, str(y), ha='center', va='bottom', fontsize=9)
        
        # Quarterly trend (last 12 quarters)
        quarters = sorted(time_data['by_quarter'].keys())[-12:]
        quarterly_counts = [time_data['by_quarter'][q] for q in quarters]
        
        axes[1].bar(range(len(quarters)), quarterly_counts, color=sns.color_palette("viridis", len(quarters)))
        axes[1].set_title('Secret Leaks Over Time (Quarterly - Last 12 Quarters)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Quarter', fontsize=12)
        axes[1].set_ylabel('Number of Leaks', fontsize=12)
        axes[1].set_xticks(range(len(quarters)))
        axes[1].set_xticklabels(quarters, rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(quarterly_counts):
            axes[1].text(i, v, str(v), ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, '04_leak_over_time.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_path}")
    
    def plot_top_languages(self):
        """Plot top languages with leaks"""
        print("📊 Creating top languages chart...")
        
        lang_data = self.data['top_languages']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Top 10 languages by leak count
        leaks = list(lang_data['leaks'].items())[:10]
        langs, counts = zip(*leaks) if leaks else ([], [])
        
        colors = sns.color_palette("husl", len(langs))
        bars = ax1.barh(range(len(langs)), counts, color=colors)
        ax1.set_yticks(range(len(langs)))
        ax1.set_yticklabels(langs)
        ax1.set_xlabel('Number of Leaks', fontsize=12)
        ax1.set_title('Top 10 Programming Languages with Secret Leaks', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax1.text(count, i, f' {count}', va='center', fontsize=10)
        
        # Leak rate by language (leaks per repo)
        leak_rates = []
        for lang in langs:
            leak_count = lang_data['leaks'].get(lang, 0)
            repo_count = lang_data['repos'].get(lang, 1)
            rate = leak_count / repo_count
            leak_rates.append(rate)
        
        colors2 = sns.color_palette("rocket", len(langs))
        bars2 = ax2.barh(range(len(langs)), leak_rates, color=colors2)
        ax2.set_yticks(range(len(langs)))
        ax2.set_yticklabels(langs)
        ax2.set_xlabel('Leaks per Repository', fontsize=12)
        ax2.set_title('Leak Rate by Programming Language', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()
        
        # Add value labels
        for i, (bar, rate) in enumerate(zip(bars2, leak_rates)):
            ax2.text(rate, i, f' {rate:.2f}', va='center', fontsize=10)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, '05_top_languages.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_path}")
    
    def plot_repo_characteristics(self):
        """Plot comparison of repos with and without leaks"""
        print("📊 Creating repo characteristics comparison chart...")
        
        char_data = self.data['repo_characteristics']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Repository Characteristics: With Leaks vs Without Leaks', 
                     fontsize=16, fontweight='bold')
        
        categories = ['with_leaks', 'without_leaks']
        labels = ['With Leaks', 'Without Leaks']
        colors = ['#E63946', '#2A9D8F']
        
        # Count comparison
        counts = [char_data[cat]['count'] for cat in categories]
        axes[0, 0].bar(labels, counts, color=colors)
        axes[0, 0].set_title('Number of Repositories', fontweight='bold')
        axes[0, 0].set_ylabel('Count')
        for i, v in enumerate(counts):
            axes[0, 0].text(i, v, str(v), ha='center', va='bottom')
        
        # Average stars
        avg_stars = [char_data[cat]['avg_stars'] for cat in categories]
        axes[0, 1].bar(labels, avg_stars, color=colors)
        axes[0, 1].set_title('Average Stars', fontweight='bold')
        axes[0, 1].set_ylabel('Stars')
        for i, v in enumerate(avg_stars):
            axes[0, 1].text(i, v, f'{v:.1f}', ha='center', va='bottom')
        
        # Average size
        avg_size = [char_data[cat]['avg_size'] for cat in categories]
        axes[0, 2].bar(labels, avg_size, color=colors)
        axes[0, 2].set_title('Average Size (KB)', fontweight='bold')
        axes[0, 2].set_ylabel('Size (KB)')
        for i, v in enumerate(avg_size):
            axes[0, 2].text(i, v, f'{v:.1f}', ha='center', va='bottom')
        
        # Average forks
        avg_forks = [char_data[cat]['avg_forks'] for cat in categories]
        axes[1, 0].bar(labels, avg_forks, color=colors)
        axes[1, 0].set_title('Average Forks', fontweight='bold')
        axes[1, 0].set_ylabel('Forks')
        for i, v in enumerate(avg_forks):
            axes[1, 0].text(i, v, f'{v:.1f}', ha='center', va='bottom')
        
        # Fork percentage
        fork_pct = [char_data[cat]['is_fork_pct'] for cat in categories]
        axes[1, 1].bar(labels, fork_pct, color=colors)
        axes[1, 1].set_title('Percentage of Forked Repos', fontweight='bold')
        axes[1, 1].set_ylabel('Percentage (%)')
        axes[1, 1].set_ylim(0, 100)
        for i, v in enumerate(fork_pct):
            axes[1, 1].text(i, v, f'{v:.1f}%', ha='center', va='bottom')
        
        # Archived percentage
        archived_pct = [char_data[cat]['is_archived_pct'] for cat in categories]
        axes[1, 2].bar(labels, archived_pct, color=colors)
        axes[1, 2].set_title('Percentage of Archived Repos', fontweight='bold')
        axes[1, 2].set_ylabel('Percentage (%)')
        axes[1, 2].set_ylim(0, 100)
        for i, v in enumerate(archived_pct):
            axes[1, 2].text(i, v, f'{v:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, '06_repo_characteristics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_path}")
    
    def plot_response_times(self):
        """Plot issue response and close times"""
        print("📊 Creating response times chart...")
        
        if 'response_times' not in self.data or not self.data['response_times']:
            print("⚠️  No response time data available")
            return
        
        rt_data = self.data['response_times']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Response times
        if 'response' in rt_data:
            metrics = ['avg_hours', 'median_hours', 'min_hours', 'max_hours']
            labels = ['Average', 'Median', 'Minimum', 'Maximum']
            values = [rt_data['response'][m] / 24 for m in metrics]  # Convert to days
            
            colors = sns.color_palette("Blues_r", len(values))
            bars = axes[0].bar(labels, values, color=colors)
            axes[0].set_title('Issue Response Time (Days)', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Days', fontsize=11)
            
            for bar, val in zip(bars, values):
                axes[0].text(bar.get_x() + bar.get_width()/2, val,
                           f'{val:.1f}', ha='center', va='bottom')
        
        # Close times
        if 'close' in rt_data:
            metrics = ['avg_hours', 'median_hours', 'min_hours', 'max_hours']
            labels = ['Average', 'Median', 'Minimum', 'Maximum']
            values = [rt_data['close'][m] / 24 for m in metrics]  # Convert to days
            
            colors = sns.color_palette("Reds_r", len(values))
            bars = axes[1].bar(labels, values, color=colors)
            axes[1].set_title('Issue Close Time (Days)', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Days', fontsize=11)
            
            for bar, val in zip(bars, values):
                axes[1].text(bar.get_x() + bar.get_width()/2, val,
                           f'{val:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, '07_response_times.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_path}")
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("\n" + "=" * 80)
        print("📊 GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        self.load_analysis_data()
        
        self.plot_issue_types()
        self.plot_leak_by_repo_size()
        self.plot_leak_by_stars()
        self.plot_leak_over_time()
        self.plot_top_languages()
        self.plot_repo_characteristics()
        self.plot_response_times()
        
        print("\n" + "=" * 80)
        print(f"✅ All visualizations saved to: {self.output_dir}")
        print("=" * 80)


def main():
    visualizer = LeakVisualizer()
    visualizer.generate_all_visualizations()


if __name__ == '__main__':
    main()
