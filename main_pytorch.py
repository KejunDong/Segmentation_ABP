#!/usr/bin/env python3
"""
Main script for U-Net training pipeline using PyTorch for diastolic chest compression segmentation.
This script runs the complete pipeline from data preprocessing to model evaluation.
"""

import os
import sys
import argparse
import numpy as np
import torch
from datetime import datetime

# Import our modules
from data_preprocessing import preprocess_data, normalize_signals, create_cross_validation_splits, save_preprocessed_data
from unet_model_pytorch import UNet1DPyTorch, UNetTrainer, create_data_loaders
from training_pipeline_simple import TrainingPipelineSimple as TrainingPipelinePyTorch
# from data_augmentation import DataAugmentation, create_balanced_dataset  # Removed - file deleted

def main():
    """Main function to run the complete pipeline."""
    parser = argparse.ArgumentParser(description='U-Net Training Pipeline for 4-Class Signal Segmentation (PyTorch)')
    parser.add_argument('--data_path', type=str, default='/labs/hulab/Kejun/07_ABP_Seg/sample_data.npy',
                       help='Path to the sample data file')
    parser.add_argument('--output_dir', type=str, default='/labs/hulab/Kejun/07_ABP_Seg/results_pytorch',
                       help='Output directory for results')
    parser.add_argument('--use_all_segments', action='store_true', default=True,
                       help='Use all segments (default: True)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--use_augmentation', action='store_true',
                       help='Use data augmentation')
    parser.add_argument('--augmentation_factor', type=int, default=2,
                       help='Data augmentation factor')
    parser.add_argument('--balance_dataset', action='store_true',
                       help='Balance the dataset by oversampling positive samples')
    # Removed test_mode - always use full dataset for proper CV
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, or auto)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("CUDA not available, using CPU")
    else:
        device = args.device
        print(f"Using specified device: {device}")
    
    print("="*80)
    print("U-NET TRAINING PIPELINE FOR 4-CLASS SIGNAL SEGMENTATION (PYTORCH)")
    print("="*80)
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Use all segments: {args.use_all_segments}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print(f"Use augmentation: {args.use_augmentation}")
    print(f"Balance dataset: {args.balance_dataset}")
    print("="*80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Step 1: Data Preprocessing
        print("\n" + "="*60)
        print("STEP 1: DATA PREPROCESSING")
        print("="*60)
        
        # Use full dataset for proper 10-fold cross-validation
        print("Using full dataset for 10-fold cross-validation...")
        
        X, y, segment_info = preprocess_data(args.data_path, use_all_segments=args.use_all_segments)
        
        # Normalize signals
        print("Normalizing signals...")
        X = normalize_signals(X)
        
        # Data augmentation (disabled - file removed)
        # if args.use_augmentation:
        #     print(f"Applying data augmentation with factor {args.augmentation_factor}...")
        #     aug = DataAugmentation()
        #     X, y = aug.create_augmented_dataset(X, y, args.augmentation_factor)
        
        # Balance dataset (disabled - file removed)
        # if args.balance_dataset:
        #     print("Balancing dataset...")
        #     X, y = create_balanced_dataset(X, y, target_positive_ratio=0.3)
        
        # Create cross-validation splits
        print("Creating 10-fold cross-validation splits...")
        cv_splits = create_cross_validation_splits(X, y, n_folds=10)
        
        # Save preprocessed data
        preprocessed_dir = os.path.join(args.output_dir, 'preprocessed_data')
        save_preprocessed_data(X, y, segment_info, preprocessed_dir)
        np.save(os.path.join(preprocessed_dir, 'cv_splits.npy'), cv_splits)
        
        print(f"Preprocessed data saved to {preprocessed_dir}")
        print(f"Total segments for training: {len(X)}")
        print(f"Cross-validation folds: {len(cv_splits)}")
        
        # Step 2: Model Training
        print("\n" + "="*60)
        print("STEP 2: MODEL TRAINING")
        print("="*60)
        
        # Create training pipeline
        training_dir = os.path.join(args.output_dir, 'training')
        pipeline = TrainingPipelinePyTorch(preprocessed_dir, training_dir, num_classes=4)
        
        # Run cross-validation
        print("Starting 10-fold cross-validation training...")
        cv_results = pipeline.run_cross_validation(epochs=args.epochs, batch_size=args.batch_size, device=device)
        
        # Step 3: Save Final Results
        print("\n" + "="*60)
        print("STEP 3: SAVING FINAL RESULTS")
        print("="*60)
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'parameters': vars(args),
            'data_info': {
                'total_segments': len(X),
                'num_classes': 4,
                'cv_folds': len(cv_splits)
            },
            'device_used': device
        }
        
        summary_file = os.path.join(args.output_dir, 'training_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("U-NET 4-CLASS SEGMENTATION TRAINING SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write("TRAINING PARAMETERS:\n")
            f.write("-" * 30 + "\n")
            for key, value in vars(args).items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\nDATA INFORMATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total segments: {summary['data_info']['total_segments']}\n")
            f.write(f"Number of classes: {summary['data_info']['num_classes']}\n")
            f.write(f"Cross-validation folds: {summary['data_info']['cv_folds']}\n")
            
            f.write(f"\nSYSTEM INFORMATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Device used: {summary['device_used']}\n")
            f.write(f"Timestamp: {summary['timestamp']}\n")
            
            f.write(f"\nCLASS MAPPING:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Class 0: nothing (background)\n")
            f.write(f"Class 1: diastolic chest compression (target class)\n")
            f.write(f"Class 2: systolic events (chest compression + normal heartbeat)\n")
            f.write(f"Class 3: diastolic normal heartbeat\n")
        
        print(f"Training summary saved to {summary_file}")
        
        # Print final results
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        
        if cv_results:
            print("Cross-Validation Results (Mean ± Std):")
            # Calculate overall metrics
            metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'dice_coefficient', 'iou']
            for metric in metrics_names:
                values = [fold['val_metrics'].get(metric, 0) for fold in cv_results if 'val_metrics' in fold]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
        
        print(f"\nAll results saved to: {args.output_dir}")
        print("Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\nError in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
