#!/usr/bin/env python3
"""
Custom Visualizations Script

Creates specific visualizations from the trend analysis:
1. Pie chart for issue types by counts
2. First figure of total leak by size
3. Leaks over time from year 2016 onwards
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class CustomVisualizer:
    """Create custom visualizations for leak analysis"""
    
    def __init__(self, analysis_dir: str = 'analysis', output_dir: str = 'visualizations'):
        self.analysis_dir = analysis_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
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
    
    def plot_issue_types_pie(self):
        """Plot only pie chart for issue type distribution"""
        print("📊 Creating issue type pie chart...")
        
        issue_types = self.data['issue_types']['counts']
        
        plt.figure(figsize=(10, 8))
        
        # Pie chart
        colors = sns.color_palette("husl", len(issue_types))
        wedges, texts, autotexts = plt.pie(
            issue_types.values(),
            labels=issue_types.keys(),
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        plt.title('Issue Type Distribution', fontsize=30, fontweight='bold')
        
        # Make percentage text bold and white
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(20)
        
        # Make label text bold
        for text in texts:
            text.set_fontsize(20)
            text.set_fontweight('bold')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'leak_vs_issue_type.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_path}")
    
    def plot_total_leak_by_size(self):
        """Plot only the first figure - total leaks by repository size"""
        print("📊 Creating total leaks by size chart...")
        
        size_data = self.data['leak_by_size']
        
        plt.figure(figsize=(10, 7))
        
        sizes = list(size_data.keys())
        total_leaks = [size_data[s]['total_leaks'] for s in sizes]
        
        colors = sns.color_palette("viridis", len(sizes))
        bars = plt.bar(sizes, total_leaks, color=colors)

        plt.title('Total Leaks by Repository Size', fontsize=30, fontweight='bold')
        plt.xlabel('Repository Size', fontsize=20)
        plt.ylabel('Total Leaks', fontsize=20)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(total_leaks):
            plt.text(i, v, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'leak_vs_size.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_path}")
    
    def plot_leak_over_time_from_2016(self):
        """Plot leak occurrence over time starting from 2016"""
        print("📊 Creating leak over time chart (from 2016)...")
        
        time_data = self.data['leak_over_time']
        
        plt.figure(figsize=(14, 7))
        
        # Filter yearly data from 2016 onwards
        years = sorted([int(y) for y in time_data['by_year'].keys() if int(y) >= 2016])
        yearly_counts = [time_data['by_year'][str(y)] for y in years]
        
        plt.plot(years, yearly_counts, marker='o', linewidth=2.5, markersize=10, 
                color='#2E86AB', label='Leaks per Year')
        plt.fill_between(years, yearly_counts, alpha=0.3, color='#2E86AB')
        
        plt.title('Secret Leaks Over Time (2016 - Present)', fontsize=30, fontweight='bold')
        plt.xlabel('Year', fontsize=20)
        plt.ylabel('Number of Leaks', fontsize=20)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for x, y in zip(years, yearly_counts):
            plt.text(x, y, str(y), ha='center', va='bottom', fontsize=20, fontweight='bold')
        
        # Set x-axis to show all years
        plt.xticks(years, rotation=45)

        plt.legend(fontsize=20)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'leak_vs_time.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_path}")
    
    def generate_custom_visualizations(self):
        """Generate all custom visualizations"""
        print("\n" + "=" * 80)
        print("📊 GENERATING CUSTOM VISUALIZATIONS")
        print("=" * 80)
        
        self.load_analysis_data()
        
        # 1. Pie chart for issue types
        self.plot_issue_types_pie()
        
        # 2. First figure of total leak by size
        self.plot_total_leak_by_size()
        
        # 3. Leaks over time from 2016
        self.plot_leak_over_time_from_2016()
        
        print("\n" + "=" * 80)
        print(f"✅ All custom visualizations saved to: {self.output_dir}")
        print("=" * 80)


def main():
    visualizer = CustomVisualizer()
    visualizer.generate_custom_visualizations()


if __name__ == '__main__':
    main()
