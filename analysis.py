import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io, metrics
from pathlib import Path
import re
from tqdm import tqdm

# Set plot style
plt.style.use('ggplot')
sns.set_context("notebook", font_scale=1.2)

# Define the base directory and paths
base_dir = "/Users/sparsh/Desktop/slide_results"  # Change this to your base directory path
original_dir = os.path.join(base_dir, "original")
noisy_dir = os.path.join(base_dir, "noisy")

# List of original and noisy images
original_images = ["9.png", "145.png", "258.png", "274.png", "374.png"]
noisy_images = ["9_poisson.png", "145_rician.png", "258_speckle.png", "274_gaussian.png", "374_salt_pepper.png"]

# Map original image numbers to noisy image names for easy lookup
original_to_noisy = {
    "9": "9_poisson.png",
    "145": "145_rician.png",
    "258": "258_speckle.png",
    "274": "274_gaussian.png",
    "374": "374_salt_pepper.png"
}

# Map image numbers to noise types
noise_types = {
    "9": "poisson",
    "145": "rician",
    "258": "speckle",
    "274": "gaussian",
    "374": "salt_pepper"
}

# List of metrics to compute
def calculate_metrics(original_img, denoised_img):
    """Calculate various image quality metrics"""
    # Ensure images are in the range [0, 1] for metric calculation
    if original_img.max() > 1.0:
        original_img = original_img / 255.0
    if denoised_img.max() > 1.0:
        denoised_img = denoised_img / 255.0
        
    metrics_dict = {
        'mse': metrics.mean_squared_error(original_img, denoised_img),
        'psnr': metrics.peak_signal_noise_ratio(original_img, denoised_img),
        'ssim': metrics.structural_similarity(original_img, denoised_img, 
                                             data_range=1.0,
                                             channel_axis=2 if len(original_img.shape) > 2 else None),
        'nrmse': metrics.normalized_root_mse(original_img, denoised_img)
    }
    
    return metrics_dict

def extract_method_and_img_number(filename):
    """Extract method and image number from filename"""
    # Pattern to extract from filenames like MeanFilter_3x3_9_poisson.png
    pattern = r"(.+?)_(\d+)(?:_([a-z_]+))?.png"
    match = re.match(pattern, filename)
    
    if match:
        method = match.group(1)
        img_number = match.group(2)
        return method, img_number
    else:
        # For original or noisy images with simple naming
        pattern_simple = r"(\d+)(?:_([a-z_]+))?.png"
        match_simple = re.match(pattern_simple, filename)
        if match_simple:
            img_number = match_simple.group(1)
            return "original" if match_simple.group(2) is None else "noisy", img_number
        return None, None

def main():
    # Find all method directories
    method_dirs = [d for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d)) 
                  and d not in ["original", "noisy"]]
    
    print(f"Found method directories: {method_dirs}")
    
    # Create a list to store results
    results = []
    
    # Process noisy images first (baseline)
    print("Processing noisy images (baseline)...")
    for img_num in original_images:
        img_base = os.path.splitext(img_num)[0]
        noisy_img_name = original_to_noisy[img_base]
        
        original_path = os.path.join(original_dir, img_num)
        noisy_path = os.path.join(noisy_dir, noisy_img_name)
        
        if os.path.exists(original_path) and os.path.exists(noisy_path):
            original_img = io.imread(original_path)
            noisy_img = io.imread(noisy_path)
            
            # Calculate metrics
            metrics_dict = calculate_metrics(original_img, noisy_img)
            
            # Add to results
            results.append({
                'img_name': img_base,
                'method': 'noisy',
                'noise_type': noise_types[img_base],
                **metrics_dict
            })
    
    # Process each method directory
    print("Processing denoising method results...")
    for method_dir in method_dirs:
        dir_path = os.path.join(base_dir, method_dir)
        for filename in os.listdir(dir_path):
            if filename.endswith('.png'):
                method, img_number = extract_method_and_img_number(filename)
                
                if not img_number:
                    # Try to extract from directory name if not in filename
                    method = method_dir
                    pattern = r"(\d+)(?:_([a-z_]+))?.png"
                    match = re.match(pattern, filename)
                    if match:
                        img_number = match.group(1)
                    else:
                        continue
                
                # Find corresponding original image
                original_path = os.path.join(original_dir, f"{img_number}.png")
                
                if os.path.exists(original_path):
                    original_img = io.imread(original_path)
                    denoised_img = io.imread(os.path.join(dir_path, filename))
                    
                    # Calculate metrics
                    metrics_dict = calculate_metrics(original_img, denoised_img)
                    
                    # Add to results
                    results.append({
                        'img_name': img_number,
                        'method': method,
                        'noise_type': noise_types[img_number],
                        **metrics_dict
                    })
    
    # Convert results to DataFrame and save to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(base_dir, "denoising_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")
    
    # Create visualizations
    create_visualizations(df)

def create_visualizations(df):
    """Create various visualizations for the denoising metrics"""
    output_dir = os.path.join(base_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set a better color palette - using tab20 which has more distinct colors
    methods = df['method'].unique()
    
    # Essential methods that should ALWAYS be included in visualizations
    required_methods = ['noisy', 'cGAN_denoiser', 'OptimizedKSVD_p8_n160_s5', 'OptimizedKSVD_p5_n100_s4']
    
    # Find top methods based on average SSIM (excluding the required ones we'll add later)
    other_methods = [m for m in methods if m not in required_methods]
    top_methods_df = df[df['method'].isin(other_methods)].groupby('method')['ssim'].mean().sort_values(ascending=False).head(8)
    
    # Combine top methods with required methods
    top_methods = top_methods_df.index.tolist()
    for method in required_methods:
        if method in methods and method not in top_methods:
            top_methods.append(method)
    
    # Create a filtered dataframe with only top methods for some plots
    df_top_methods = df[df['method'].isin(top_methods)]
    
    # Choose a color palette with distinct colors
    if len(top_methods) <= 10:
        palette = sns.color_palette("tab10", len(top_methods))
    else:
        palette = sns.color_palette("tab20", len(top_methods))
    
    method_colors = dict(zip(top_methods, palette))
    
    # 1. Improved Scatter plot showing averages only (one dot per method)
    plt.figure(figsize=(14, 10))
    
    # Calculate average SSIM and PSNR for each method
    method_avgs = df.groupby('method')[['ssim', 'psnr']].mean().reset_index()
    
    # Filter for top methods only
    top_method_avgs = method_avgs[method_avgs['method'].isin(top_methods)]
    
    # Sort by SSIM for color gradient
    top_method_avgs = top_method_avgs.sort_values('ssim', ascending=False)
    
    # Plot average points
    for i, row in top_method_avgs.iterrows():
        plt.scatter(row['psnr'], row['ssim'], 
                  label=row['method'], s=150, alpha=0.9,
                  color=method_colors.get(row['method'], plt.cm.tab10(i % 10)),
                  edgecolor='black', linewidth=1.0)
    
    plt.xlabel('PSNR (dB)', fontsize=14)
    plt.ylabel('SSIM', fontsize=14)
    plt.title('Average SSIM vs PSNR for Different Denoising Methods', fontsize=16)
    
    # Add method labels directly next to points
    for i, row in top_method_avgs.iterrows():
        plt.annotate(row['method'], 
                   (row['psnr'], row['ssim']),
                   xytext=(7, 0), 
                   textcoords='offset points',
                   fontsize=11,
                   alpha=0.8)
    
    # Improve legend - in this case we can hide it since we have direct labels
    plt.legend().set_visible(False)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_ssim_vs_psnr.png'), dpi=300, bbox_inches='tight')
    
    # 2. Improved Bar plot for top methods by noise type - using SSIM instead of PSNR
    plt.figure(figsize=(16, 10))
    
    # Reshape data for grouped bar chart
    pivot_data = df_top_methods.pivot_table(
        index='noise_type', 
        columns='method', 
        values='ssim', 
        aggfunc='mean'
    ).reset_index()
    
    # Convert to long format for seaborn
    melted_data = pd.melt(pivot_data, id_vars=['noise_type'], 
                        value_vars=[col for col in pivot_data.columns if col != 'noise_type'],
                        var_name='method', value_name='ssim')
    
    # Sort by ssim within each noise type
    order = melted_data.groupby('noise_type')['ssim'].mean().sort_values(ascending=False).index
    
    # Create the bar plot with improved aesthetics
    ax = sns.barplot(x='noise_type', y='ssim', hue='method', data=melted_data, 
                   palette=method_colors, order=order)
    
    plt.xlabel('Noise Type', fontsize=14)
    plt.ylabel('SSIM', fontsize=14)
    plt.title('SSIM by Noise Type and Denoising Method', fontsize=16)
    
    # Improve legend placement
    plt.legend(title='Method', fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Improve axis formatting
    plt.xticks(rotation=30, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ssim_by_noise_type.png'), dpi=300, bbox_inches='tight')
    
    # 3. Improved Heatmap - focusing on top methods only for readability, using SSIM
    pivot_ssim = df_top_methods.pivot_table(
        index='method', 
        columns='noise_type',
        values='ssim',
        aggfunc='mean'
    )
    
    # Sort methods by average performance (SSIM)
    pivot_ssim = pivot_ssim.reindex(pivot_ssim.mean(axis=1).sort_values(ascending=False).index)
    
    plt.figure(figsize=(14, max(8, len(top_methods) * 0.4)))
    sns.heatmap(pivot_ssim, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=.5)
    plt.title('SSIM Heatmap: Methods vs. Noise Types', fontsize=16)
    plt.yticks(rotation=0, fontsize=10)  # Make method names horizontal and readable
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ssim_heatmap.png'), dpi=300, bbox_inches='tight')
    
    # 4. Top method for each noise type (better visualization) - using SSIM
    noise_types_list = df['noise_type'].unique()
    
    # Calculate the top 3 methods for each noise type based on SSIM
    top_methods_by_noise = {}
    for noise in noise_types_list:
        noise_data = df[df['noise_type'] == noise]
        # Get top 3 methods
        top3 = noise_data.sort_values('ssim', ascending=False).head(3)
        top_methods_by_noise[noise] = list(zip(top3['method'], top3['ssim']))
    
    # Create a summary table
    rows = []
    for noise, methods_with_scores in top_methods_by_noise.items():
        for i, (method, score) in enumerate(methods_with_scores):
            rank = i + 1
            rows.append({
                'Noise Type': noise,
                'Rank': rank,
                'Method': method,
                'SSIM': score
            })
    
    summary_df = pd.DataFrame(rows)
    
    # Save the summary to CSV
    summary_df.to_csv(os.path.join(output_dir, 'best_methods_summary.csv'), index=False)
    
    # Create a visually appealing table plot
    plt.figure(figsize=(15, len(noise_types_list) * 1.5))
    
    # Create a table visualization
    noise_types = sorted(noise_types_list)
    table_data = []
    
    for noise in noise_types:
        noise_methods = [m for n, m, _ in sorted([(n, m, p) for n, ms in top_methods_by_noise.items() 
                                        for i, (m, p) in enumerate(ms) if n == noise], 
                                     key=lambda x: (x[0], x[2]), reverse=True)]
        noise_scores = [f"{p:.3f}" for n, _, p in sorted([(n, m, p) for n, ms in top_methods_by_noise.items() 
                                             for i, (m, p) in enumerate(ms) if n == noise], 
                                          key=lambda x: (x[0], x[2]), reverse=True)]
        table_data.append([noise] + [f"{m}\n({s})" for m, s in zip(noise_methods, noise_scores)])
    
    column_labels = ['Noise Type', '1st Place', '2nd Place', '3rd Place']
    
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    table = plt.table(cellText=table_data, colLabels=column_labels, loc='center',
                    cellLoc='center', colWidths=[0.2, 0.3, 0.3, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Color the header row
    for i, key in enumerate(column_labels):
        cell = table[0, i]
        cell.set_text_props(fontproperties=plt.matplotlib.font_manager.FontProperties(weight='bold'))
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white')
    
    # Color the 1st place cells
    for i in range(len(noise_types)):
        cell = table[i+1, 1]  # +1 because of the header row
        cell.set_facecolor('#C6E0B4')  # Light green
    
    # Color the 2nd place cells
    for i in range(len(noise_types)):
        cell = table[i+1, 2]
        cell.set_facecolor('#FFE699')  # Light yellow
    
    # Color the 3rd place cells
    for i in range(len(noise_types)):
        cell = table[i+1, 3]
        cell.set_facecolor('#F8CBAD')  # Light orange
    
    plt.title('Top 3 Methods by Noise Type (with SSIM scores)', fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_methods_by_noise.png'), dpi=300, bbox_inches='tight')
    
    # 5. Method comparison boxplot - focus on SSIM - ENSURE REQUIRED METHODS ARE INCLUDED
    plt.figure(figsize=(16, 10))
    
    # Sort methods by median SSIM (higher is better)
    order = df_top_methods.groupby('method')['ssim'].median().sort_values(ascending=False).index
    
    # Create a better boxplot with correct parameter usage
    ax = sns.boxplot(
        x='method', 
        y='ssim', 
        hue='method',  # Add hue parameter to fix deprecation warning
        data=df_top_methods, 
        palette=method_colors,
        order=order,
        width=0.6,
        fliersize=5,
        legend=False  # Hide legend since it's redundant with x-axis
    )
    
    # Add swarmplot for individual data points
    sns.swarmplot(
        x='method', 
        y='ssim', 
        data=df_top_methods, 
        color='black', 
        alpha=0.5,
        size=4,
        order=order
    )
    
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('SSIM', fontsize=14)
    plt.title('SSIM Distribution by Method', fontsize=16)
    
    # Improve readability of method names
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Add reference line for noisy baseline if it exists in the data
    if 'noisy' in df_top_methods['method'].values:
        noisy_median = df_top_methods[df_top_methods['method'] == 'noisy']['ssim'].median()
        plt.axhline(y=noisy_median, color='red', linestyle='--', alpha=0.7, 
                  label=f'Noisy Baseline: {noisy_median:.4f}')
        plt.legend()
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ssim_boxplot.png'), dpi=300, bbox_inches='tight')
    
    # 6. NEW: Performance radar chart for top methods and required methods - using SSIM
    # Use the same top_methods list that includes required methods
    radar_df = df[df['method'].isin(top_methods)]
    
    # Get average performance by noise type
    radar_data = []
    for method in top_methods:
        method_data = radar_df[radar_df['method'] == method]
        noise_avgs = {}
        for noise in noise_types_list:
            noise_data = method_data[method_data['noise_type'] == noise]
            if not noise_data.empty:
                # Normalize SSIM to 0-1 scale across all methods for this noise type
                min_ssim = df[df['noise_type'] == noise]['ssim'].min()
                max_ssim = df[df['noise_type'] == noise]['ssim'].max()
                if max_ssim > min_ssim:
                    noise_avgs[noise] = (noise_data['ssim'].mean() - min_ssim) / (max_ssim - min_ssim)
                else:
                    noise_avgs[noise] = 0.5
        radar_data.append({**{'method': method}, **noise_avgs})
    
    radar_df = pd.DataFrame(radar_data)
    
    # Create the radar chart
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of variables
    categories = noise_types_list
    N = len(categories)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw the y-axis labels (0.0 to 1.0)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Create a custom color palette for the radar chart
    radar_colors = sns.color_palette("tab10", len(top_methods))
    
    # Plot each method
    for i, method in enumerate(top_methods):
        if radar_df[radar_df['method'] == method].empty:
            continue
            
        values = radar_df[radar_df['method'] == method].iloc[0].drop('method').values.tolist()
        values += values[:1]  # Close the loop
        
        # Plot the values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=method, 
              color=radar_colors[i])
        ax.fill(angles, values, alpha=0.1, color=radar_colors[i])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Performance Across Noise Types\n(Normalized SSIM, higher is better)', size=15, y=1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_chart_ssim.png'), dpi=300, bbox_inches='tight')
    
    # 7. Performance improvement percentage over noisy baseline - using SSIM
    if 'noisy' in df['method'].values:
        # Calculate baseline performance
        baseline = df[df['method'] == 'noisy'].groupby('noise_type')['ssim'].mean()
        
        # Calculate improvement percentage for each method over baseline
        improvement_data = []
        
        for method in methods:
            if method == 'noisy':
                continue
                
            method_data = df[df['method'] == method]
            
            for noise in noise_types_list:
                noise_data = method_data[method_data['noise_type'] == noise]
                if not noise_data.empty:
                    avg_ssim = noise_data['ssim'].mean()
                    baseline_ssim = baseline[noise]
                    # For SSIM, the baseline might be close to 0, causing extreme percentage values
                    # Use a more robust improvement calculation
                    if baseline_ssim > 0.01:  # Only if baseline is not too small
                        improvement = ((avg_ssim - baseline_ssim) / baseline_ssim) * 100
                    else:
                        improvement = (avg_ssim - baseline_ssim) * 100  # Absolute difference * 100
                    
                    improvement_data.append({
                        'method': method,
                        'noise_type': noise,
                        'improvement_pct': improvement
                    })
        
        improvement_df = pd.DataFrame(improvement_data)
        
        # Get top methods by average improvement
        top_improving_methods = improvement_df.groupby('method')['improvement_pct'].mean().sort_values(ascending=False).head(10).index.tolist()
        
        # Add specific methods requested by user if they're not already in the top methods
        for method in required_methods:
            if method in methods and method != 'noisy' and method not in top_improving_methods:
                top_improving_methods.append(method)
        
        # Filter to only include top improving methods
        plot_df = improvement_df[improvement_df['method'].isin(top_improving_methods)]
        
        # Create the plot
        plt.figure(figsize=(16, 10))
        
        # Create a pivot table for the heatmap
        pivot_improvement = plot_df.pivot_table(
            index='method', 
            columns='noise_type',
            values='improvement_pct', 
            aggfunc='mean'
        )
        
        # Sort methods by average improvement
        pivot_improvement = pivot_improvement.reindex(pivot_improvement.mean(axis=1).sort_values(ascending=False).index)
        
        # Create the heatmap with appropriate color scale
        vmax = min(200, pivot_improvement.values.max())  # Cap at 200% to avoid extreme values
        vmin = max(-50, pivot_improvement.values.min())  # Cap at -50%
        
        sns.heatmap(pivot_improvement, annot=True, cmap='RdYlGn', fmt='.1f', 
                  linewidths=.5, center=0, vmin=vmin, vmax=vmax)
        
        plt.title('SSIM Improvement (%) Over Noisy Baseline', fontsize=16)
        plt.yticks(rotation=0, fontsize=10)
        plt.xticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ssim_improvement_heatmap.png'), dpi=300, bbox_inches='tight')
    
    # 8. Create visual comparison of specific methods
    # Create comparison line chart for SSIM across noise types
    plt.figure(figsize=(14, 8))
    
    # Use the same top_methods list to ensure required methods are included
    comparison_methods = top_methods
    
    # Create a filtered dataframe
    comparison_df = df[df['method'].isin(comparison_methods)]
    
    # Calculate average SSIM by method and noise type
    comparison_pivot = comparison_df.pivot_table(
        index='noise_type',
        columns='method',
        values='ssim',
        aggfunc='mean'
    )
    
    # Plot the comparison
    ax = comparison_pivot.plot(
        kind='bar',
        figsize=(16, 10),
        width=0.8
    )
    
    plt.xlabel('Noise Type', fontsize=14)
    plt.ylabel('SSIM (higher is better)', fontsize=14)
    plt.title('Comparison of Denoising Methods by Noise Type', fontsize=16)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title='Method', fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_comparison_by_noise.png'), dpi=300, bbox_inches='tight')
    
    # 9. Create line chart to compare methods across noise types
    plt.figure(figsize=(14, 10))
    
    # Get average SSIM by method and noise type
    method_by_noise = df[df['method'].isin(comparison_methods)].groupby(['method', 'noise_type'])['ssim'].mean().reset_index()
    
    # Create the line plot with different marker styles for each method
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|']
    
    # Create a custom color palette
    line_colors = sns.color_palette("tab10", len(comparison_methods))
    
    # Create a figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # First subplot: Line chart
    for i, method in enumerate(comparison_methods):
        method_data = method_by_noise[method_by_noise['method'] == method]
        ax1.plot(method_data['noise_type'], method_data['ssim'], 
               marker=markers[i % len(markers)], 
               markersize=10, 
               linewidth=2, 
               label=method,
               color=line_colors[i % len(line_colors)])
    
    ax1.set_xlabel('Noise Type', fontsize=14)
    ax1.set_ylabel('SSIM', fontsize=14)
    ax1.set_title('Method Performance Across Noise Types', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(title='Method', fontsize=10, title_fontsize=12)
    
    # Second subplot: Method ranking by noise type
    # Create a DataFrame with rankings
    ranking_df = pd.DataFrame()
    
    for noise in df['noise_type'].unique():
        noise_data = df[df['method'].isin(comparison_methods) & (df['noise_type'] == noise)]
        noise_ranking = noise_data.groupby('method')['ssim'].mean().rank(ascending=False)
        ranking_df[noise] = noise_ranking
    
    # Calculate average rank across all noise types
    ranking_df['Average'] = ranking_df.mean(axis=1)
    ranking_df = ranking_df.sort_values('Average')
    
    # Create heatmap of rankings
    sns.heatmap(ranking_df.drop('Average', axis=1), annot=True, cmap='YlGnBu_r', fmt='.0f', 
              linewidths=.5, ax=ax2, cbar_kws={'label': 'Rank (lower is better)'})
    
    ax2.set_title('Method Ranking by Noise Type (1 = best)', fontsize=16)
    ax2.set_xlabel('Noise Type', fontsize=14)
    ax2.set_ylabel('Method', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_performance_lines_and_ranks.png'), dpi=300, bbox_inches='tight')
    
    # 10. NEW: Generate a report of the best methods based on average SSIM across all noise types
    # Calculate the average SSIM across all noise types for each method
    method_avg_ssim = df.groupby('method')['ssim'].mean().sort_values(ascending=False).reset_index()
    
    # Get the top 10 methods
    top10_methods = method_avg_ssim.head(10)
    
    # Create a bar plot of the best methods
    plt.figure(figsize=(16, 10))
    
    ax = sns.barplot(x='method', y='ssim', data=top10_methods, palette='viridis')
    
    # Add value labels on top of each bar
    for i, v in enumerate(top10_methods['ssim']):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
    
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Average SSIM', fontsize=14)
    plt.title('Top 10 Methods by Average SSIM Across All Noise Types', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_methods_overall.png'), dpi=300, bbox_inches='tight')
    
    # Create a summary report CSV
    # Best method overall
    best_overall = method_avg_ssim.iloc[0]['method']
    best_overall_ssim = method_avg_ssim.iloc[0]['ssim']
    
    # Best method for each noise type
    best_by_noise = {}
    for noise in noise_types_list:
        noise_data = df[df['noise_type'] == noise]
        best_method = noise_data.groupby('method')['ssim'].mean().sort_values(ascending=False).index[0]
        best_ssim = noise_data[noise_data['method'] == best_method]['ssim'].mean()
        best_by_noise[noise] = (best_method, best_ssim)
    
    # Generate report
    report_rows = []
    
    # Overall best method
    report_rows.append({
        'Category': 'Best Overall',
        'Noise Type': 'All',
        'Best Method': best_overall,
        'SSIM': best_overall_ssim
    })
    
    # Best method for each noise type
    for noise, (method, ssim) in best_by_noise.items():
        report_rows.append({
            'Category': 'Best by Noise Type',
            'Noise Type': noise,
            'Best Method': method,
            'SSIM': ssim
        })
    
    # Create DataFrame and save to CSV
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(os.path.join(output_dir, 'best_methods_report.csv'), index=False)
    
    # Create a visually appealing summary table
    plt.figure(figsize=(14, 6))
    
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    # Create table data
    table_data = []
    table_data.append(['All Noise Types', f"{best_overall} ({best_overall_ssim:.3f})"])
    for noise, (method, ssim) in best_by_noise.items():
        table_data.append([noise, f"{method} ({ssim:.3f})"])
    
    column_labels = ['Noise Type', 'Best Method (SSIM)']
    
    table = plt.table(cellText=table_data, colLabels=column_labels, loc='center',
                    cellLoc='center', colWidths=[0.3, 0.7])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Color the header row
    for i, key in enumerate(column_labels):
        cell = table[0, i]
        cell.set_text_props(fontproperties=plt.matplotlib.font_manager.FontProperties(weight='bold'))
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white')
    
    # Color the overall best method row
    cell = table[1, 0]
    cell.set_facecolor('#C6E0B4')  # Light green
    cell = table[1, 1]
    cell.set_facecolor('#C6E0B4')  # Light green
    
    plt.title('Best Denoising Methods Summary', fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_methods_summary_table.png'), dpi=300, bbox_inches='tight')
    
    print(f"Enhanced visualizations saved to {output_dir}")
    print(f"Overall best method: {best_overall} with average SSIM: {best_overall_ssim:.3f}") 
    df_top_methods = df[df['method'].isin(top_methods)]
    
    # Choose a color palette with distinct colors
    if len(top_methods) <= 10:
        palette = sns.color_palette("tab10", len(top_methods))
    else:
        palette = sns.color_palette("tab20", len(top_methods))
    
    method_colors = dict(zip(top_methods, palette))
    
    # 1. Improved Scatter plot showing averages only (one dot per method)
    plt.figure(figsize=(14, 10))
    
    # Calculate average SSIM and PSNR for each method
    method_avgs = df.groupby('method')[['ssim', 'psnr']].mean().reset_index()
    
    # Filter for top methods only
    top_method_avgs = method_avgs[method_avgs['method'].isin(top_methods)]
    
    # Sort by SSIM for color gradient
    top_method_avgs = top_method_avgs.sort_values('ssim', ascending=False)
    
    # Plot average points
    for i, row in top_method_avgs.iterrows():
        plt.scatter(row['psnr'], row['ssim'], 
                  label=row['method'], s=150, alpha=0.9,
                  color=method_colors.get(row['method'], plt.cm.tab10(i % 10)),
                  edgecolor='black', linewidth=1.0)
    
    plt.xlabel('PSNR (dB)', fontsize=14)
    plt.ylabel('SSIM', fontsize=14)
    plt.title('Average SSIM vs PSNR for Different Denoising Methods', fontsize=16)
    
    # Add method labels directly next to points
    for i, row in top_method_avgs.iterrows():
        plt.annotate(row['method'], 
                   (row['psnr'], row['ssim']),
                   xytext=(7, 0), 
                   textcoords='offset points',
                   fontsize=11,
                   alpha=0.8)
    
    # Improve legend - in this case we can hide it since we have direct labels
    plt.legend().set_visible(False)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_ssim_vs_psnr.png'), dpi=300, bbox_inches='tight')
    
    # 2. Improved Bar plot for top methods by noise type - using SSIM instead of PSNR
    plt.figure(figsize=(16, 10))
    
    # Reshape data for grouped bar chart
    pivot_data = df_top_methods.pivot_table(
        index='noise_type', 
        columns='method', 
        values='ssim', 
        aggfunc='mean'
    ).reset_index()
    
    # Convert to long format for seaborn
    melted_data = pd.melt(pivot_data, id_vars=['noise_type'], 
                        value_vars=[col for col in pivot_data.columns if col != 'noise_type'],
                        var_name='method', value_name='ssim')
    
    # Sort by ssim within each noise type
    order = melted_data.groupby('noise_type')['ssim'].mean().sort_values(ascending=False).index
    
    # Create the bar plot with improved aesthetics
    ax = sns.barplot(x='noise_type', y='ssim', hue='method', data=melted_data, 
                   palette=method_colors, order=order)
    
    plt.xlabel('Noise Type', fontsize=14)
    plt.ylabel('SSIM', fontsize=14)
    plt.title('SSIM by Noise Type and Denoising Method', fontsize=16)
    
    # Improve legend placement
    plt.legend(title='Method', fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Improve axis formatting
    plt.xticks(rotation=30, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ssim_by_noise_type.png'), dpi=300, bbox_inches='tight')
    
    # 3. Improved Heatmap - focusing on top methods only for readability, using SSIM
    pivot_ssim = df_top_methods.pivot_table(
        index='method', 
        columns='noise_type',
        values='ssim',
        aggfunc='mean'
    )
    
    # Sort methods by average performance (SSIM)
    pivot_ssim = pivot_ssim.reindex(pivot_ssim.mean(axis=1).sort_values(ascending=False).index)
    
    plt.figure(figsize=(14, max(8, len(top_methods) * 0.4)))
    sns.heatmap(pivot_ssim, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=.5)
    plt.title('SSIM Heatmap: Methods vs. Noise Types', fontsize=16)
    plt.yticks(rotation=0, fontsize=10)  # Make method names horizontal and readable
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ssim_heatmap.png'), dpi=300, bbox_inches='tight')
    
    # 4. Top method for each noise type (better visualization) - using SSIM
    noise_types_list = df['noise_type'].unique()
    
    # Calculate the top 3 methods for each noise type based on SSIM
    top_methods_by_noise = {}
    for noise in noise_types_list:
        noise_data = df[df['noise_type'] == noise]
        # Get top 3 methods
        top3 = noise_data.sort_values('ssim', ascending=False).head(3)
        top_methods_by_noise[noise] = list(zip(top3['method'], top3['ssim']))
    
    # Create a summary table
    rows = []
    for noise, methods_with_scores in top_methods_by_noise.items():
        for i, (method, score) in enumerate(methods_with_scores):
            rank = i + 1
            rows.append({
                'Noise Type': noise,
                'Rank': rank,
                'Method': method,
                'SSIM': score
            })
    
    summary_df = pd.DataFrame(rows)
    
    # Save the summary to CSV
    summary_df.to_csv(os.path.join(output_dir, 'best_methods_summary.csv'), index=False)
    
    # Create a visually appealing table plot
    plt.figure(figsize=(15, len(noise_types_list) * 1.5))
    
    # Create a table visualization
    noise_types = sorted(noise_types_list)
    table_data = []
    
    for noise in noise_types:
        noise_methods = [m for n, m, _ in sorted([(n, m, p) for n, ms in top_methods_by_noise.items() 
                                        for i, (m, p) in enumerate(ms) if n == noise], 
                                     key=lambda x: (x[0], x[2]), reverse=True)]
        noise_scores = [f"{p:.3f}" for n, _, p in sorted([(n, m, p) for n, ms in top_methods_by_noise.items() 
                                             for i, (m, p) in enumerate(ms) if n == noise], 
                                          key=lambda x: (x[0], x[2]), reverse=True)]
        table_data.append([noise] + [f"{m}\n({s})" for m, s in zip(noise_methods, noise_scores)])
    
    column_labels = ['Noise Type', '1st Place', '2nd Place', '3rd Place']
    
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    table = plt.table(cellText=table_data, colLabels=column_labels, loc='center',
                    cellLoc='center', colWidths=[0.2, 0.3, 0.3, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Color the header row
    for i, key in enumerate(column_labels):
        cell = table[0, i]
        cell.set_text_props(fontproperties=plt.matplotlib.font_manager.FontProperties(weight='bold'))
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white')
    
    # Color the 1st place cells
    for i in range(len(noise_types)):
        cell = table[i+1, 1]  # +1 because of the header row
        cell.set_facecolor('#C6E0B4')  # Light green
    
    # Color the 2nd place cells
    for i in range(len(noise_types)):
        cell = table[i+1, 2]
        cell.set_facecolor('#FFE699')  # Light yellow
    
    # Color the 3rd place cells
    for i in range(len(noise_types)):
        cell = table[i+1, 3]
        cell.set_facecolor('#F8CBAD')  # Light orange
    
    plt.title('Top 3 Methods by Noise Type (with SSIM scores)', fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_methods_by_noise.png'), dpi=300, bbox_inches='tight')
    
    # 5. Method comparison boxplot - focus on SSIM - ENSURE REQUIRED METHODS ARE INCLUDED
    plt.figure(figsize=(16, 10))
    
    # Sort methods by median SSIM (higher is better)
    order = df_top_methods.groupby('method')['ssim'].median().sort_values(ascending=False).index
    
    # Create a better boxplot with correct parameter usage
    ax = sns.boxplot(
        x='method', 
        y='ssim', 
        hue='method',  # Add hue parameter to fix deprecation warning
        data=df_top_methods, 
        palette=method_colors,
        order=order,
        width=0.6,
        fliersize=5,
        legend=False  # Hide legend since it's redundant with x-axis
    )
    
    # Add swarmplot for individual data points
    sns.swarmplot(
        x='method', 
        y='ssim', 
        data=df_top_methods, 
        color='black', 
        alpha=0.5,
        size=4,
        order=order
    )
    
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('SSIM', fontsize=14)
    plt.title('SSIM Distribution by Method', fontsize=16)
    
    # Improve readability of method names
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Add reference line for noisy baseline if it exists in the data
    if 'noisy' in df_top_methods['method'].values:
        noisy_median = df_top_methods[df_top_methods['method'] == 'noisy']['ssim'].median()
        plt.axhline(y=noisy_median, color='red', linestyle='--', alpha=0.7, 
                  label=f'Noisy Baseline: {noisy_median:.4f}')
        plt.legend()
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ssim_boxplot.png'), dpi=300, bbox_inches='tight')
    
    # 6. NEW: Performance radar chart for top methods and required methods - using SSIM
    # Use the same top_methods list that includes required methods
    radar_df = df[df['method'].isin(top_methods)]
    
    # Get average performance by noise type
    radar_data = []
    for method in top_methods:
        method_data = radar_df[radar_df['method'] == method]
        noise_avgs = {}
        for noise in noise_types_list:
            noise_data = method_data[method_data['noise_type'] == noise]
            if not noise_data.empty:
                # Normalize SSIM to 0-1 scale across all methods for this noise type
                min_ssim = df[df['noise_type'] == noise]['ssim'].min()
                max_ssim = df[df['noise_type'] == noise]['ssim'].max()
                if max_ssim > min_ssim:
                    noise_avgs[noise] = (noise_data['ssim'].mean() - min_ssim) / (max_ssim - min_ssim)
                else:
                    noise_avgs[noise] = 0.5
        radar_data.append({**{'method': method}, **noise_avgs})
    
    radar_df = pd.DataFrame(radar_data)
    
    # Create the radar chart
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of variables
    categories = noise_types_list
    N = len(categories)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw the y-axis labels (0.0 to 1.0)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Create a custom color palette for the radar chart
    radar_colors = sns.color_palette("tab10", len(top_methods))
    
    # Plot each method
    for i, method in enumerate(top_methods):
        if radar_df[radar_df['method'] == method].empty:
            continue
            
        values = radar_df[radar_df['method'] == method].iloc[0].drop('method').values.tolist()
        values += values[:1]  # Close the loop
        
        # Plot the values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=method, 
              color=radar_colors[i])
        ax.fill(angles, values, alpha=0.1, color=radar_colors[i])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Performance Across Noise Types\n(Normalized SSIM, higher is better)', size=15, y=1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_chart_ssim.png'), dpi=300, bbox_inches='tight')
    
    # 7. Performance improvement percentage over noisy baseline - using SSIM
    if 'noisy' in df['method'].values:
        # Calculate baseline performance
        baseline = df[df['method'] == 'noisy'].groupby('noise_type')['ssim'].mean()
        
        # Calculate improvement percentage for each method over baseline
        improvement_data = []
        
        for method in methods:
            if method == 'noisy':
                continue
                
            method_data = df[df['method'] == method]
            
            for noise in noise_types_list:
                noise_data = method_data[method_data['noise_type'] == noise]
                if not noise_data.empty:
                    avg_ssim = noise_data['ssim'].mean()
                    baseline_ssim = baseline[noise]
                    # For SSIM, the baseline might be close to 0, causing extreme percentage values
                    # Use a more robust improvement calculation
                    if baseline_ssim > 0.01:  # Only if baseline is not too small
                        improvement = ((avg_ssim - baseline_ssim) / baseline_ssim) * 100
                    else:
                        improvement = (avg_ssim - baseline_ssim) * 100  # Absolute difference * 100
                    
                    improvement_data.append({
                        'method': method,
                        'noise_type': noise,
                        'improvement_pct': improvement
                    })
        
        improvement_df = pd.DataFrame(improvement_data)
        
        # Get top methods by average improvement
        top_improving_methods = improvement_df.groupby('method')['improvement_pct'].mean().sort_values(ascending=False).head(10).index.tolist()
        
        # Add specific methods requested by user if they're not already in the top methods
        for method in required_methods:
            if method in methods and method != 'noisy' and method not in top_improving_methods:
                top_improving_methods.append(method)
        
        # Filter to only include top improving methods
        plot_df = improvement_df[improvement_df['method'].isin(top_improving_methods)]
        
        # Create the plot
        plt.figure(figsize=(16, 10))
        
        # Create a pivot table for the heatmap
        pivot_improvement = plot_df.pivot_table(
            index='method', 
            columns='noise_type',
            values='improvement_pct', 
            aggfunc='mean'
        )
        
        # Sort methods by average improvement
        pivot_improvement = pivot_improvement.reindex(pivot_improvement.mean(axis=1).sort_values(ascending=False).index)
        
        # Create the heatmap with appropriate color scale
        vmax = min(200, pivot_improvement.values.max())  # Cap at 200% to avoid extreme values
        vmin = max(-50, pivot_improvement.values.min())  # Cap at -50%
        
        sns.heatmap(pivot_improvement, annot=True, cmap='RdYlGn', fmt='.1f', 
                  linewidths=.5, center=0, vmin=vmin, vmax=vmax)
        
        plt.title('SSIM Improvement (%) Over Noisy Baseline', fontsize=16)
        plt.yticks(rotation=0, fontsize=10)
        plt.xticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ssim_improvement_heatmap.png'), dpi=300, bbox_inches='tight')
    
    # 8. Create visual comparison of specific methods
    # Create comparison line chart for SSIM across noise types
    plt.figure(figsize=(14, 8))
    
    # Use the same top_methods list to ensure required methods are included
    comparison_methods = top_methods
    
    # Create a filtered dataframe
    comparison_df = df[df['method'].isin(comparison_methods)]
    
    # Calculate average SSIM by method and noise type
    comparison_pivot = comparison_df.pivot_table(
        index='noise_type',
        columns='method',
        values='ssim',
        aggfunc='mean'
    )
    
    # Plot the comparison
    ax = comparison_pivot.plot(
        kind='bar',
        figsize=(16, 10),
        width=0.8
    )
    
    plt.xlabel('Noise Type', fontsize=14)
    plt.ylabel('SSIM (higher is better)', fontsize=14)
    plt.title('Comparison of Denoising Methods by Noise Type', fontsize=16)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title='Method', fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_comparison_by_noise.png'), dpi=300, bbox_inches='tight')
    
    # 9. Create line chart to compare methods across noise types
    plt.figure(figsize=(14, 10))
    
    # Get average SSIM by method and noise type
    method_by_noise = df[df['method'].isin(comparison_methods)].groupby(['method', 'noise_type'])['ssim'].mean().reset_index()
    
    # Create the line plot with different marker styles for each method
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|']
    
    # Create a custom color palette
    line_colors = sns.color_palette("tab10", len(comparison_methods))
    
    # Create a figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # First subplot: Line chart
    for i, method in enumerate(comparison_methods):
        method_data = method_by_noise[method_by_noise['method'] == method]
        ax1.plot(method_data['noise_type'], method_data['ssim'], 
               marker=markers[i % len(markers)], 
               markersize=10, 
               linewidth=2, 
               label=method,
               color=line_colors[i % len(line_colors)])
    
    ax1.set_xlabel('Noise Type', fontsize=14)
    ax1.set_ylabel('SSIM', fontsize=14)
    ax1.set_title('Method Performance Across Noise Types', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(title='Method', fontsize=10, title_fontsize=12)
    
    # Second subplot: Method ranking by noise type
    # Create a DataFrame with rankings
    ranking_df = pd.DataFrame()
    
    for noise in df['noise_type'].unique():
        noise_data = df[df['method'].isin(comparison_methods) & (df['noise_type'] == noise)]
        noise_ranking = noise_data.groupby('method')['ssim'].mean().rank(ascending=False)
        ranking_df[noise] = noise_ranking
    
    # Calculate average rank across all noise types
    ranking_df['Average'] = ranking_df.mean(axis=1)
    ranking_df = ranking_df.sort_values('Average')
    
    # Create heatmap of rankings
    sns.heatmap(ranking_df.drop('Average', axis=1), annot=True, cmap='YlGnBu_r', fmt='.0f', 
              linewidths=.5, ax=ax2, cbar_kws={'label': 'Rank (lower is better)'})
    
    ax2.set_title('Method Ranking by Noise Type (1 = best)', fontsize=16)
    ax2.set_xlabel('Noise Type', fontsize=14)
    ax2.set_ylabel('Method', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_performance_lines_and_ranks.png'), dpi=300, bbox_inches='tight')
    
    # 10. NEW: Generate a report of the best methods based on average SSIM across all noise types
    # Calculate the average SSIM across all noise types for each method
    method_avg_ssim = df.groupby('method')['ssim'].mean().sort_values(ascending=False).reset_index()
    
    # Get the top 10 methods
    top10_methods = method_avg_ssim.head(10)
    
    # Create a bar plot of the best methods
    plt.figure(figsize=(16, 10))
    
    ax = sns.barplot(x='method', y='ssim', data=top10_methods, palette='viridis')
    
    # Add value labels on top of each bar
    for i, v in enumerate(top10_methods['ssim']):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
    
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Average SSIM', fontsize=14)
    plt.title('Top 10 Methods by Average SSIM Across All Noise Types', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_methods_overall.png'), dpi=300, bbox_inches='tight')
    
    # Create a summary report CSV
    # Best method overall
    best_overall = method_avg_ssim.iloc[0]['method']
    best_overall_ssim = method_avg_ssim.iloc[0]['ssim']
    
    # Best method for each noise type
    best_by_noise = {}
    for noise in noise_types_list:
        noise_data = df[df['noise_type'] == noise]
        best_method = noise_data.groupby('method')['ssim'].mean().sort_values(ascending=False).index[0]
        best_ssim = noise_data[noise_data['method'] == best_method]['ssim'].mean()
        best_by_noise[noise] = (best_method, best_ssim)
    
    # Generate report
    report_rows = []
    
    # Overall best method
    report_rows.append({
        'Category': 'Best Overall',
        'Noise Type': 'All',
        'Best Method': best_overall,
        'SSIM': best_overall_ssim
    })
    
    # Best method for each noise type
    for noise, (method, ssim) in best_by_noise.items():
        report_rows.append({
            'Category': 'Best by Noise Type',
            'Noise Type': noise,
            'Best Method': method,
            'SSIM': ssim
        })
    
    # Create DataFrame and save to CSV
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(os.path.join(output_dir, 'best_methods_report.csv'), index=False)
    
    # Create a visually appealing summary table
    plt.figure(figsize=(14, 6))
    
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    # Create table data
    table_data = []
    table_data.append(['All Noise Types', f"{best_overall} ({best_overall_ssim:.3f})"])
    for noise, (method, ssim) in best_by_noise.items():
        table_data.append([noise, f"{method} ({ssim:.3f})"])
    
    column_labels = ['Noise Type', 'Best Method (SSIM)']
    
    table = plt.table(cellText=table_data, colLabels=column_labels, loc='center',
                    cellLoc='center', colWidths=[0.3, 0.7])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Color the header row
    for i, key in enumerate(column_labels):
        cell = table[0, i]
        cell.set_text_props(fontproperties=plt.matplotlib.font_manager.FontProperties(weight='bold'))
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white')
    
    # Color the overall best method row
    cell = table[1, 0]
    cell.set_facecolor('#C6E0B4')  # Light green
    cell = table[1, 1]
    cell.set_facecolor('#C6E0B4')  # Light green
    
    plt.title('Best Denoising Methods Summary', fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_methods_summary_table.png'), dpi=300, bbox_inches='tight')
    
    print(f"Enhanced visualizations saved to {output_dir}")
    print(f"Overall best method: {best_overall} with average SSIM: {best_overall_ssim:.3f}")
    # Create a filtered dataframe with only top methods for some plots
    df_top_methods = df[df['method'].isin(top_methods)]
    
    # Choose a color palette with distinct colors
    if len(top_methods) <= 10:
        palette = sns.color_palette("tab10", len(top_methods))
    else:
        palette = sns.color_palette("tab20", len(top_methods))
    
    method_colors = dict(zip(top_methods, palette))
    
    # 1. Improved Scatter plot showing averages only (one dot per method)
    plt.figure(figsize=(14, 10))
    
    # Calculate average SSIM and PSNR for each method
    method_avgs = df.groupby('method')[['ssim', 'psnr']].mean().reset_index()
    
    # Filter for top methods only
    top_method_avgs = method_avgs[method_avgs['method'].isin(top_methods)]
    
    # Sort by SSIM for color gradient
    top_method_avgs = top_method_avgs.sort_values('ssim', ascending=False)
    
    # Plot average points
    for i, row in top_method_avgs.iterrows():
        plt.scatter(row['psnr'], row['ssim'], 
                  label=row['method'], s=150, alpha=0.9,
                  color=method_colors.get(row['method'], plt.cm.tab10(i % 10)),
                  edgecolor='black', linewidth=1.0)
    
    plt.xlabel('PSNR (dB)', fontsize=14)
    plt.ylabel('SSIM', fontsize=14)
    plt.title('Average SSIM vs PSNR for Different Denoising Methods', fontsize=16)
    
    # Add method labels directly next to points
    for i, row in top_method_avgs.iterrows():
        plt.annotate(row['method'], 
                   (row['psnr'], row['ssim']),
                   xytext=(7, 0), 
                   textcoords='offset points',
                   fontsize=11,
                   alpha=0.8)
    
    # Improve legend - in this case we can hide it since we have direct labels
    plt.legend().set_visible(False)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_ssim_vs_psnr.png'), dpi=300, bbox_inches='tight')
    
    # 2. Improved Bar plot for top methods by noise type - using SSIM instead of PSNR
    plt.figure(figsize=(16, 10))
    
    # Reshape data for grouped bar chart
    pivot_data = df_top_methods.pivot_table(
        index='noise_type', 
        columns='method', 
        values='ssim', 
        aggfunc='mean'
    ).reset_index()
    
    # Convert to long format for seaborn
    melted_data = pd.melt(pivot_data, id_vars=['noise_type'], 
                        value_vars=[col for col in pivot_data.columns if col != 'noise_type'],
                        var_name='method', value_name='ssim')
    
    # Sort by ssim within each noise type
    order = melted_data.groupby('noise_type')['ssim'].mean().sort_values(ascending=False).index
    
    # Create the bar plot with improved aesthetics
    ax = sns.barplot(x='noise_type', y='ssim', hue='method', data=melted_data, 
                   palette=method_colors, order=order)
    
    plt.xlabel('Noise Type', fontsize=14)
    plt.ylabel('SSIM', fontsize=14)
    plt.title('SSIM by Noise Type and Denoising Method', fontsize=16)
    
    # Improve legend placement
    plt.legend(title='Method', fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Improve axis formatting
    plt.xticks(rotation=30, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ssim_by_noise_type.png'), dpi=300, bbox_inches='tight')
    
    # 3. Improved Heatmap - focusing on top methods only for readability, using SSIM
    pivot_ssim = df_top_methods.pivot_table(
        index='method', 
        columns='noise_type',
        values='ssim',
        aggfunc='mean'
    )
    
    # Sort methods by average performance (SSIM)
    pivot_ssim = pivot_ssim.reindex(pivot_ssim.mean(axis=1).sort_values(ascending=False).index)
    
    plt.figure(figsize=(14, max(8, len(top_methods) * 0.4)))
    sns.heatmap(pivot_ssim, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=.5)
    plt.title('SSIM Heatmap: Methods vs. Noise Types', fontsize=16)
    plt.yticks(rotation=0, fontsize=10)  # Make method names horizontal and readable
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ssim_heatmap.png'), dpi=300, bbox_inches='tight')
    
    # 4. Top method for each noise type (better visualization) - using SSIM
    noise_types_list = df['noise_type'].unique()
    
    # Calculate the top 3 methods for each noise type based on SSIM
    top_methods_by_noise = {}
    for noise in noise_types_list:
        noise_data = df[df['noise_type'] == noise]
        # Get top 3 methods
        top3 = noise_data.sort_values('ssim', ascending=False).head(3)
        top_methods_by_noise[noise] = list(zip(top3['method'], top3['ssim']))
    
    # Create a summary table
    rows = []
    for noise, methods_with_scores in top_methods_by_noise.items():
        for i, (method, score) in enumerate(methods_with_scores):
            rank = i + 1
            rows.append({
                'Noise Type': noise,
                'Rank': rank,
                'Method': method,
                'SSIM': score
            })
    
    summary_df = pd.DataFrame(rows)
    
    # Save the summary to CSV
    summary_df.to_csv(os.path.join(output_dir, 'best_methods_summary.csv'), index=False)
    
    # Create a visually appealing table plot
    plt.figure(figsize=(15, len(noise_types_list) * 1.5))
    
    # Create a table visualization
    noise_types = sorted(noise_types_list)
    table_data = []
    
    for noise in noise_types:
        noise_methods = [m for n, m, _ in sorted([(n, m, p) for n, ms in top_methods_by_noise.items() 
                                        for i, (m, p) in enumerate(ms) if n == noise], 
                                     key=lambda x: (x[0], x[2]), reverse=True)]
        noise_scores = [f"{p:.3f}" for n, _, p in sorted([(n, m, p) for n, ms in top_methods_by_noise.items() 
                                             for i, (m, p) in enumerate(ms) if n == noise], 
                                          key=lambda x: (x[0], x[2]), reverse=True)]
        table_data.append([noise] + [f"{m}\n({s})" for m, s in zip(noise_methods, noise_scores)])
    
    column_labels = ['Noise Type', '1st Place', '2nd Place', '3rd Place']
    
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    table = plt.table(cellText=table_data, colLabels=column_labels, loc='center',
                    cellLoc='center', colWidths=[0.2, 0.3, 0.3, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Color the header row
    for i, key in enumerate(column_labels):
        cell = table[0, i]
        cell.set_text_props(fontproperties=plt.matplotlib.font_manager.FontProperties(weight='bold'))
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white')
    
    # Color the 1st place cells
    for i in range(len(noise_types)):
        cell = table[i+1, 1]  # +1 because of the header row
        cell.set_facecolor('#C6E0B4')  # Light green
    
    # Color the 2nd place cells
    for i in range(len(noise_types)):
        cell = table[i+1, 2]
        cell.set_facecolor('#FFE699')  # Light yellow
    
    # Color the 3rd place cells
    for i in range(len(noise_types)):
        cell = table[i+1, 3]
        cell.set_facecolor('#F8CBAD')  # Light orange
    
    plt.title('Top 3 Methods by Noise Type (with SSIM scores)', fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_methods_by_noise.png'), dpi=300, bbox_inches='tight')
    
    # 5. Method comparison boxplot - focus on SSIM
    plt.figure(figsize=(16, 10))
    
    # Sort methods by median SSIM (higher is better)
    order = df_top_methods.groupby('method')['ssim'].median().sort_values(ascending=False).index
    
    # Create a better boxplot with correct parameter usage
    ax = sns.boxplot(
        x='method', 
        y='ssim', 
        hue='method',  # Add hue parameter to fix deprecation warning
        data=df_top_methods, 
        palette=method_colors,
        order=order,
        width=0.6,
        fliersize=5,
        legend=False  # Hide legend since it's redundant with x-axis
    )
    
    # Add swarmplot for individual data points
    sns.swarmplot(
        x='method', 
        y='ssim', 
        data=df_top_methods, 
        color='black', 
        alpha=0.5,
        size=4,
        order=order
    )

if __name__ == "__main__":
    main()