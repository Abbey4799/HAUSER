import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_correlations(df):
    """Calculate correlations between automatic metrics and human ratings"""
    # Calculate average human ratings
    human_ratings = (df['label1_q'] + df['label2_q'] + df['label3_q']) / 3
    
    # Define metrics to analyze
    metrics = {
        'BLEU2': df['BLEU2'],
        'rouge2': df['rouge2'], 
        'meteor': df['meteor'],
        'BERTS_large': df['BERTS_large'],
        'Perplexity': df['fluency'],
        'Quality': df['relevance_KB']*(3/6) + df['consistency_mnli']*(2/6) + df['consistency_emo']*(1/6)
    }
    
    # Calculate correlations
    results = {}
    for name, metric in metrics.items():
        pearson = pearsonr(metric, human_ratings)
        spearman = spearmanr(metric, human_ratings)
        results[name] = {
            'pearson': pearson[0],
            'pearson_p': pearson[1],
            'spearman': spearman[0],
            'spearman_p': spearman[1]
        }
    
    return results, metrics, human_ratings

def plot_correlations(metrics, human_ratings):
    """Create correlation plots"""
    # Set up the plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 10))
    
    # Create subplots for each metric
    for idx, (name, metric) in enumerate(metrics.items(), 1):
        plt.subplot(2, 3, idx)
        sns.regplot(x=metric, y=human_ratings, scatter_kws={'alpha':0.5})
        plt.title(f'{name}\nr={pearsonr(metric, human_ratings)[0]:.3f}')
        plt.xlabel('Automatic Score')
        plt.ylabel('Human Rating')
    
    plt.tight_layout()
    plt.savefig('correlation_plots.pdf')
    plt.show()

def main():
    # Load data
    df = pd.read_csv('../data/human_annotated.csv')
    
    # Calculate correlations
    results, metrics, human_ratings = calculate_correlations(df)
    
    # Print results
    print("\nCorrelation Results:")
    print("-" * 50)
    for metric, scores in results.items():
        print(f"\n{metric}:")
        print(f"Pearson: r={scores['pearson']:.3f} (p={scores['pearson_p']:.3f})")
        print(f"Spearman: ρ={scores['spearman']:.3f} (p={scores['spearman_p']:.3f})")
    
    # Create visualization
    plot_correlations(metrics, human_ratings)

if __name__ == "__main__":
    main()
