#!/usr/bin/env python3
"""
Script to generate overall results from individual fold results
"""

import json
import numpy as np
import os
from datetime import datetime

def load_fold_results(results_dir):
    """Load all fold results from JSON files."""
    fold_results = []
    
    for i in range(1, 11):  # 10 folds
        fold_file = os.path.join(results_dir, f'fold_{i}_results.json')
        if os.path.exists(fold_file):
            with open(fold_file, 'r') as f:
                fold_data = json.load(f)
                fold_results.append(fold_data)
                print(f"Loaded fold {i} results")
        else:
            print(f"Fold {i} results not found")
    
    return fold_results

def calculate_overall_statistics(fold_results):
    """Calculate overall statistics across all folds."""
    if not fold_results:
        return {}
    
    # Extract metrics from all folds
    metrics_names = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'macro_dice', 'macro_iou']
    overall_stats = {}
    
    for metric in metrics_names:
        values = [fold['val_metrics'][metric] for fold in fold_results if metric in fold['val_metrics']]
        if values:
            overall_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
    
    return overall_stats

def calculate_per_class_statistics(fold_results):
    """Calculate per-class statistics across all folds."""
    if not fold_results or 'per_class_precision' not in fold_results[0]['val_metrics']:
        return {}
    
    per_class_metrics = ['per_class_precision', 'per_class_recall', 'per_class_f1', 'per_class_dice', 'per_class_iou']
    class_names = ['nothing', 'diastolic_CC', 'systolic_CC_NH', 'diastolic_NH']
    
    per_class_stats = {}
    
    for metric in per_class_metrics:
        per_class_stats[metric] = {}
        for class_idx, class_name in enumerate(class_names):
            values = [fold['val_metrics'][metric][class_idx] for fold in fold_results if metric in fold['val_metrics']]
            if values:
                per_class_stats[metric][class_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
    
    return per_class_stats

def save_overall_results(overall_stats, per_class_stats, output_file):
    """Save overall results to file."""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("U-NET 4-CLASS SEGMENTATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OVERALL CROSS-VALIDATION RESULTS:\n")
        f.write("-" * 50 + "\n")
        
        if overall_stats:
            for metric, stats in overall_stats.items():
                f.write(f"{metric.upper()}:\n")
                f.write(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
                f.write(f"  Values: {[f'{v:.4f}' for v in stats['values']]}\n\n")
        else:
            f.write("Overall statistics not available.\n\n")
        
        # Per-class results
        if per_class_stats:
            f.write("\nPER-CLASS RESULTS (Mean ± Std across folds):\n")
            f.write("-" * 50 + "\n")
            
            class_names = ['nothing', 'diastolic_CC', 'systolic_CC_NH', 'diastolic_NH']
            
            for class_name in class_names:
                f.write(f"\n{class_name.upper()}:\n")
                for metric in ['per_class_precision', 'per_class_recall', 'per_class_f1', 'per_class_dice', 'per_class_iou']:
                    if metric in per_class_stats and class_name in per_class_stats[metric]:
                        stats = per_class_stats[metric][class_name]
                        metric_name = metric.replace('per_class_', '').upper()
                        f.write(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
        
        f.write(f"\nTRAINING DETAILS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total folds: {len(fold_results) if 'fold_results' in locals() else 0}\n")
        f.write(f"Number of classes: 4\n")
        f.write(f"Class names: nothing, diastolic_CC, systolic_CC_NH, diastolic_NH\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def save_detailed_results(fold_results, output_file):
    """Save detailed per-fold results to file."""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DETAILED PER-FOLD RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for i, fold in enumerate(fold_results):
            f.write(f"FOLD {i+1}:\n")
            f.write("-" * 30 + "\n")
            
            # Overall metrics
            f.write("Overall Metrics:\n")
            for metric in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'macro_dice', 'macro_iou']:
                if metric in fold['val_metrics']:
                    f.write(f"  {metric}: {fold['val_metrics'][metric]:.4f}\n")
            
            # Per-class metrics
            if 'per_class_precision' in fold['val_metrics']:
                f.write("\nPer-Class Metrics:\n")
                class_names = ['nothing', 'diastolic_CC', 'systolic_CC_NH', 'diastolic_NH']
                metrics = ['per_class_precision', 'per_class_recall', 'per_class_f1', 'per_class_dice', 'per_class_iou']
                
                for class_name in class_names:
                    f.write(f"  {class_name}:\n")
                    for metric in metrics:
                        if metric in fold['val_metrics']:
                            class_idx = class_names.index(class_name)
                            value = fold['val_metrics'][metric][class_idx]
                            metric_name = metric.replace('per_class_', '').upper()
                            f.write(f"    {metric_name}: {value:.4f}\n")
            
            f.write("\n" + "="*50 + "\n\n")

def main():
    results_dir = "/labs/hulab/Kejun/07_ABP_Seg/results_pytorch/training/results"
    
    print("Loading fold results...")
    fold_results = load_fold_results(results_dir)
    
    if not fold_results:
        print("No fold results found!")
        return
    
    print(f"Loaded {len(fold_results)} fold results")
    
    print("Calculating overall statistics...")
    overall_stats = calculate_overall_statistics(fold_results)
    
    print("Calculating per-class statistics...")
    per_class_stats = calculate_per_class_statistics(fold_results)
    
    print("Saving overall results...")
    overall_file = os.path.join(results_dir, 'overall_results.txt')
    save_overall_results(overall_stats, per_class_stats, overall_file)
    
    print("Saving detailed results...")
    detailed_file = os.path.join(results_dir, 'detailed_results.txt')
    save_detailed_results(fold_results, detailed_file)
    
    print("Results generation complete!")
    
    # Print summary
    print("\n" + "="*60)
    print("OVERALL CROSS-VALIDATION RESULTS")
    print("="*60)
    
    for metric, stats in overall_stats.items():
        print(f"{metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")

if __name__ == "__main__":
    main()


