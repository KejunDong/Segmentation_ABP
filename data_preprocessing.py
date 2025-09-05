import numpy as np
from sklearn.model_selection import KFold
import os

def preprocess_data(data_path, use_all_segments=True):
    """
    Preprocess the data for 4-class U-Net training.
    Class mapping: 0->0, 1->1, 2->2, 3->3, 4->2 (combine 2&4)
    
    Args:
        data_path: Path to sample_data.npy
        use_all_segments: If True, use all segments; if False, only segments with labels 1,2,3,4
    
    Returns:
        X: Input signals (n_segments, 400, 1)
        y: 4-class labels (n_segments, 400, 1) - 0: nothing, 1: diastolic CC, 2: systolic (CC+NH), 3: diastolic NH
        segment_info: Information about each segment
    """
    print("Loading data...")
    data = np.load(data_path)
    input_data = data[0]  # Shape: (6800, 400)
    label_data = data[1]  # Shape: (6800, 400)
    
    print(f"Original data shape: {input_data.shape}")
    print(f"Original label shape: {label_data.shape}")
    
    if use_all_segments:
        # Use all segments
        target_segments = list(range(label_data.shape[0]))
        print(f"Using all {len(target_segments)} segments")
    else:
        # Filter segments that contain any of the target labels (1,2,3,4)
        target_segments = []
        for i in range(label_data.shape[0]):
            if any(label in label_data[i, :] for label in [1, 2, 3, 4]):
                target_segments.append(i)
        print(f"Found {len(target_segments)} segments containing labels 1,2,3,4")
    
    # Extract target segments
    X = input_data[target_segments]  # Shape: (n_segments, 400)
    y_original = label_data[target_segments]  # Shape: (n_segments, 400)
    
    # Create 4-class labels: 0->0, 1->1, 2->2, 3->3, 4->2
    y = y_original.copy().astype(np.int32)
    y[y_original == 4] = 2  # Combine labels 2 and 4 into class 2
    
    # Reshape for U-Net (add channel dimension)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Shape: (n_segments, 400, 1)
    y = y.reshape(y.shape[0], y.shape[1], 1)  # Shape: (n_segments, 400, 1)
    
    # Create segment information
    segment_info = {
        'original_indices': target_segments,
        'n_segments': len(target_segments),
        'class_mapping': {0: 'nothing', 1: 'diastolic_CC', 2: 'systolic_CC_NH', 3: 'diastolic_NH'},
        'n_classes': 4
    }
    
    print(f"Preprocessed data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution:")
    for class_id, class_name in segment_info['class_mapping'].items():
        count = np.sum(y == class_id)
        percentage = count / (y.shape[0] * y.shape[1]) * 100
        print(f"  Class {class_id} ({class_name}): {count} points ({percentage:.2f}%)")
    
    return X, y, segment_info

def normalize_signals(X):
    """
    Normalize signals to have zero mean and unit variance.
    """
    # Normalize each signal individually
    X_normalized = np.zeros_like(X)
    for i in range(X.shape[0]):
        signal = X[i, :, 0]
        if np.std(signal) > 0:
            X_normalized[i, :, 0] = (signal - np.mean(signal)) / np.std(signal)
        else:
            X_normalized[i, :, 0] = signal
    
    return X_normalized

def create_train_test_split(X, y, test_size=0.3, random_state=42):
    """
    Create 70/30 train/test split.
    Each segment is treated as one patient.
    """
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    
    print(f"Train/Test Split: Train={len(X_train)} ({len(X_train)/len(X)*100:.1f}%), Test={len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

def create_cross_validation_splits(X, y, n_folds=10, random_state=42):
    """
    Create 10-fold cross-validation splits.
    Each segment is treated as one patient.
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    cv_splits = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        cv_splits.append({
            'fold': fold + 1,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val
        })
        
        print(f"Fold {fold + 1}: Train={len(train_idx)}, Val={len(val_idx)}")
    
    return cv_splits

def save_preprocessed_data(X, y, segment_info, output_dir):
    """
    Save preprocessed data for later use.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)
    np.save(os.path.join(output_dir, 'segment_info.npy'), segment_info)
    
    print(f"Preprocessed data saved to {output_dir}")

def load_preprocessed_data(output_dir):
    """
    Load preprocessed data.
    """
    X = np.load(os.path.join(output_dir, 'X.npy'))
    y = np.load(os.path.join(output_dir, 'y.npy'))
    segment_info = np.load(os.path.join(output_dir, 'segment_info.npy'), allow_pickle=True).item()
    
    return X, y, segment_info

if __name__ == "__main__":
    # Preprocess data
    data_path = "/labs/hulab/Kejun/07_ABP_Seg/sample_data.npy"
    output_dir = "/labs/hulab/Kejun/07_ABP_Seg/preprocessed_data"
    
    print("Preprocessing data for 4-class U-Net training...")
    X, y, segment_info = preprocess_data(data_path, use_all_segments=True)
    
    # Normalize signals
    print("Normalizing signals...")
    X = normalize_signals(X)
    
    # Create cross-validation splits
    print("Creating 10-fold cross-validation splits...")
    cv_splits = create_cross_validation_splits(X, y, n_folds=10)
    
    # Save preprocessed data
    save_preprocessed_data(X, y, segment_info, output_dir)
    
    # Save CV splits
    np.save(os.path.join(output_dir, 'cv_splits.npy'), cv_splits)
    
    print("Data preprocessing complete!")
    print(f"Total segments for training: {len(X)}")
    print(f"Cross-validation folds: {len(cv_splits)}")
