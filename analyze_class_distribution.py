#!/usr/bin/env python3
"""
Script to analyze the detailed class distribution in the dataset
"""

import numpy as np

def analyze_class_distribution(data_path):
    """Analyze detailed class distribution."""
    print("Loading data...")
    data = np.load(data_path)
    input_data = data[0]  # Shape: (6800, 400)
    label_data = data[1]  # Shape: (6800, 400)
    
    print(f"Data shape: {input_data.shape}")
    print(f"Label shape: {label_data.shape}")
    
    # Analyze original labels
    print("\nOriginal Label Distribution:")
    unique_labels, counts = np.unique(label_data, return_counts=True)
    total_points = label_data.size
    
    for label, count in zip(unique_labels, counts):
        percentage = count / total_points * 100
        print(f"  Label {label}: {count:,} points ({percentage:.2f}%)")
    
    # Analyze after 4-class mapping (0->0, 1->1, 2->2, 3->3, 4->2)
    print("\nAfter 4-class mapping (combining labels 2 and 4):")
    y_mapped = label_data.copy()
    y_mapped[y_mapped == 4] = 2  # Combine labels 2 and 4 into class 2
    
    unique_labels_mapped, counts_mapped = np.unique(y_mapped, return_counts=True)
    
    class_names = {0: 'nothing', 1: 'diastolic_CC', 2: 'systolic_CC_NH', 3: 'diastolic_NH'}
    
    for label, count in zip(unique_labels_mapped, counts_mapped):
        percentage = count / total_points * 100
        class_name = class_names.get(label, f'unknown_{label}')
        print(f"  Class {label} ({class_name}): {count:,} points ({percentage:.2f}%)")
    
    # Analyze per-segment distribution
    print("\nPer-segment analysis:")
    segments_with_class = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for i in range(label_data.shape[0]):
        segment_labels = y_mapped[i, :]
        unique_in_segment = np.unique(segment_labels)
        
        for class_id in unique_in_segment:
            segments_with_class[class_id] += 1
    
    print("Number of segments containing each class:")
    for class_id, count in segments_with_class.items():
        class_name = class_names.get(class_id, f'unknown_{class_id}')
        percentage = count / label_data.shape[0] * 100
        print(f"  Class {class_id} ({class_name}): {count} segments ({percentage:.1f}%)")
    
    # Check for potential data leakage issues
    print("\nPotential Issues Analysis:")
    
    # Check if there are segments with only one class (might be too easy)
    single_class_segments = 0
    for i in range(label_data.shape[0]):
        segment_labels = y_mapped[i, :]
        unique_in_segment = np.unique(segment_labels)
        if len(unique_in_segment) == 1:
            single_class_segments += 1
    
    print(f"  Segments with only one class: {single_class_segments} ({single_class_segments/label_data.shape[0]*100:.1f}%)")
    
    # Check class imbalance severity
    class_counts = [counts_mapped[i] for i in range(len(unique_labels_mapped))]
    max_count = max(class_counts)
    min_count = min(class_counts)
    imbalance_ratio = max_count / min_count
    
    print(f"  Class imbalance ratio (max/min): {imbalance_ratio:.2f}")
    print(f"  Most frequent class: {max_count:,} points")
    print(f"  Least frequent class: {min_count:,} points")
    
    # Check if the majority class dominates
    majority_class_count = max(class_counts)
    majority_percentage = majority_class_count / total_points * 100
    print(f"  Majority class percentage: {majority_percentage:.2f}%")
    
    if majority_percentage > 80:
        print("  ⚠️  WARNING: Majority class dominates (>80%) - this could lead to inflated accuracy!")
    
    if imbalance_ratio > 10:
        print("  ⚠️  WARNING: Severe class imbalance (>10:1) - this could lead to biased performance!")
    
    return {
        'original_distribution': dict(zip(unique_labels, counts)),
        'mapped_distribution': dict(zip(unique_labels_mapped, counts_mapped)),
        'segments_per_class': segments_with_class,
        'single_class_segments': single_class_segments,
        'imbalance_ratio': imbalance_ratio,
        'majority_percentage': majority_percentage
    }

if __name__ == "__main__":
    data_path = "sample_data.npy"
    results = analyze_class_distribution(data_path)

