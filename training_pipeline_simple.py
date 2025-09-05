import numpy as np
import os
import json
import torch
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from unet_model_pytorch import UNet1DPyTorch, UNetTrainer, create_data_loaders
from data_preprocessing import load_preprocessed_data

class TrainingPipelineSimple:
    """
    Simplified training pipeline for U-Net with 10-fold cross-validation using PyTorch.
    No matplotlib dependencies for basic functionality.
    """
    
    def __init__(self, data_dir, output_dir, num_classes=4):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.num_classes = num_classes
        self.results = {}
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
    
    def load_data(self):
        """Load preprocessed data and CV splits."""
        print("Loading preprocessed data...")
        
        # Load main data
        X, y, segment_info = load_preprocessed_data(self.data_dir)
        
        # Load CV splits
        cv_splits = np.load(os.path.join(self.data_dir, 'cv_splits.npy'), allow_pickle=True)
        
        print(f"Loaded {len(X)} segments for training")
        print(f"Cross-validation folds: {len(cv_splits)}")
        
        return X, y, segment_info, cv_splits
    
    def train_fold(self, fold_data, fold_num, epochs=100, batch_size=32, device='cpu'):
        """Train model for one fold."""
        print(f"\n{'='*50}")
        print(f"Training Fold {fold_num}")
        print(f"{'='*50}")
        
        X_train = fold_data['X_train']
        y_train = fold_data['y_train']
        X_val = fold_data['X_val']
        y_val = fold_data['y_val']
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val, batch_size)
        
        # Create model and trainer
        model = UNet1DPyTorch(input_length=400, input_channels=1, num_classes=self.num_classes)
        trainer = UNetTrainer(model, device=device, learning_rate=0.001)
        
        # Train model
        print("Starting training...")
        history = trainer.train(train_loader, val_loader, epochs=epochs, patience=20)
        
        # Evaluate on validation set
        print("Evaluating on validation set...")
        y_pred = trainer.predict(val_loader)
        
        # Calculate metrics
        val_metrics = trainer.evaluate_metrics(y_val, y_pred)
        
        # Save model
        model_path = os.path.join(self.output_dir, 'models', f'unet_fold_{fold_num}.pth')
        torch.save(model.state_dict(), model_path)
        
        # Save fold results
        fold_results = {
            'fold': fold_num,
            'history': history,
            'val_metrics': val_metrics,
            'model_path': model_path
        }
        
        return fold_results, trainer
    
    def run_cross_validation(self, epochs=100, batch_size=32, device='cpu'):
        """Run 10-fold cross-validation."""
        print("Starting 10-fold cross-validation...")
        
        # Load data
        X, y, segment_info, cv_splits = self.load_data()
        
        # Store results for all folds
        all_fold_results = []
        
        # Train each fold
        for i, fold_data in enumerate(cv_splits):
            fold_num = i + 1
            
            try:
                fold_results, trainer = self.train_fold(
                    fold_data, fold_num, epochs, batch_size, device
                )
                all_fold_results.append(fold_results)
                
                # Save individual fold results
                fold_file = os.path.join(
                    self.output_dir, 'results', f'fold_{fold_num}_results.json'
                )
                with open(fold_file, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    fold_results_copy = fold_results.copy()
                    fold_results_copy['history'] = {
                        k: [float(x) for x in v] for k, v in fold_results['history'].items()
                    }
                    json.dump(fold_results_copy, f, indent=2)
                
                print(f"Fold {fold_num} completed successfully!")
                
            except Exception as e:
                print(f"Error in fold {fold_num}: {str(e)}")
                continue
        
        # Calculate overall statistics
        try:
            self.calculate_overall_statistics(all_fold_results)
        except Exception as e:
            print(f"Error calculating overall statistics: {str(e)}")
            # Initialize empty overall statistics if calculation fails
            self.results['overall_statistics'] = {}
        
        # Save all results
        self.save_results(all_fold_results)
        
        return all_fold_results
    
    def calculate_overall_statistics(self, all_fold_results):
        """Calculate overall statistics across all folds."""
        if not all_fold_results:
            print("No fold results to analyze!")
            return
        
        # Extract metrics from all folds
        metrics_names = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'macro_dice', 'macro_iou']
        overall_stats = {}
        
        for metric in metrics_names:
            values = [fold['val_metrics'][metric] for fold in all_fold_results if metric in fold['val_metrics']]
            if values:
                overall_stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
        
        self.results['overall_statistics'] = overall_stats
        
        # Print overall results
        print(f"\n{'='*60}")
        print("OVERALL CROSS-VALIDATION RESULTS")
        print(f"{'='*60}")
        
        for metric, stats in overall_stats.items():
            print(f"{metric.upper()}:")
            print(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print()
        
        # Print per-class results if available
        if all_fold_results and 'per_class_precision' in all_fold_results[0]['val_metrics']:
            print(f"\n{'='*60}")
            print("PER-CLASS RESULTS (Mean ± Std across folds)")
            print(f"{'='*60}")
            
            class_names = ['nothing', 'diastolic_CC', 'systolic_CC_NH', 'diastolic_NH']
            metrics = ['precision', 'recall', 'f1', 'dice', 'iou']
            
            for class_id, class_name in enumerate(class_names):
                print(f"\nClass {class_id} ({class_name}):")
                for metric in metrics:
                    metric_key = f'per_class_{metric}'
                    if metric_key in all_fold_results[0]['val_metrics']:
                        values = [fold['val_metrics'][metric_key][class_id] for fold in all_fold_results]
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    
    def save_results(self, all_fold_results):
        """Save all results to TXT files."""
        # Save overall results as TXT
        results_file = os.path.join(self.output_dir, 'results', 'overall_results.txt')
        with open(results_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("U-NET 4-CLASS SEGMENTATION RESULTS\n")
            f.write("="*80 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL CROSS-VALIDATION RESULTS:\n")
            f.write("-" * 50 + "\n")
            
            if 'overall_statistics' in self.results and self.results['overall_statistics']:
                for metric, stats in self.results['overall_statistics'].items():
                    f.write(f"{metric.upper()}:\n")
                    f.write(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                    f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
                    f.write(f"  Values: {[f'{v:.4f}' for v in stats['values']]}\n\n")
            else:
                f.write("Overall statistics not available.\n\n")
            
            # Per-class results if available
            if all_fold_results and 'per_class_precision' in all_fold_results[0]['val_metrics']:
                f.write("\nPER-CLASS RESULTS (Mean ± Std across folds):\n")
                f.write("-" * 50 + "\n")
                
                class_names = ['nothing', 'diastolic_CC', 'systolic_CC_NH', 'diastolic_NH']
                metrics = ['precision', 'recall', 'f1', 'dice', 'iou']
                
                for class_id, class_name in enumerate(class_names):
                    f.write(f"\nClass {class_id} ({class_name}):\n")
                    for metric in metrics:
                        metric_key = f'per_class_{metric}'
                        if metric_key in all_fold_results[0]['val_metrics']:
                            values = [fold['val_metrics'][metric_key][class_id] for fold in all_fold_results]
                            mean_val = np.mean(values)
                            std_val = np.std(values)
                            f.write(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}\n")
            
            # Training details
            f.write(f"\nTRAINING DETAILS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total folds: {len(all_fold_results)}\n")
            f.write(f"Number of classes: 4\n")
            f.write(f"Class names: nothing, diastolic_CC, systolic_CC_NH, diastolic_NH\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"Results saved to {results_file}")
        
        # Also save detailed per-fold results
        detailed_file = os.path.join(self.output_dir, 'results', 'detailed_results.txt')
        with open(detailed_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DETAILED PER-FOLD RESULTS\n")
            f.write("="*80 + "\n\n")
            
            for i, fold_result in enumerate(all_fold_results):
                f.write(f"FOLD {i+1}:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Validation Metrics:\n")
                
                metrics = fold_result['val_metrics']
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"  Macro Precision: {metrics['macro_precision']:.4f}\n")
                f.write(f"  Macro Recall: {metrics['macro_recall']:.4f}\n")
                f.write(f"  Macro F1: {metrics['macro_f1']:.4f}\n")
                f.write(f"  Macro Dice: {metrics['macro_dice']:.4f}\n")
                f.write(f"  Macro IoU: {metrics['macro_iou']:.4f}\n")
                
                # Per-class metrics for this fold
                if 'per_class_precision' in metrics:
                    f.write(f"\n  Per-Class Metrics:\n")
                    class_names = ['nothing', 'diastolic_CC', 'systolic_CC_NH', 'diastolic_NH']
                    for class_id, class_name in enumerate(class_names):
                        f.write(f"    Class {class_id} ({class_name}):\n")
                        f.write(f"      Precision: {metrics['per_class_precision'][class_id]:.4f}\n")
                        f.write(f"      Recall: {metrics['per_class_recall'][class_id]:.4f}\n")
                        f.write(f"      F1: {metrics['per_class_f1'][class_id]:.4f}\n")
                        f.write(f"      Dice: {metrics['per_class_dice'][class_id]:.4f}\n")
                        f.write(f"      IoU: {metrics['per_class_iou'][class_id]:.4f}\n")
                
                f.write(f"\n  Model Path: {fold_result['model_path']}\n")
                f.write("\n" + "="*50 + "\n\n")
        
        print(f"Detailed results saved to {detailed_file}")

def main():
    """Main function to run the training pipeline."""
    # Configuration
    data_dir = "/labs/hulab/Kejun/07_ABP_Seg/preprocessed_data"
    output_dir = "/labs/hulab/Kejun/07_ABP_Seg/training_results_simple"
    target_label = 1  # Diastolic chest compression
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create pipeline
    pipeline = TrainingPipelineSimple(data_dir, output_dir, target_label)
    
    # Run cross-validation
    print("Starting U-Net training pipeline with PyTorch...")
    print(f"Target label: {target_label} (Diastolic chest compression)")
    print(f"Output directory: {output_dir}")
    
    try:
        results = pipeline.run_cross_validation(epochs=50, batch_size=16, device=device)
        print("\nTraining pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    import torch
    main()
