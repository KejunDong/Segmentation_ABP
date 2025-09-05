import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

class UNet1DPyTorch(nn.Module):
    """
    1D U-Net for signal segmentation using PyTorch.
    Adapted for segmenting diastolic regions in arterial blood pressure signals.
    """
    
    def __init__(self, input_length=400, input_channels=1, num_classes=4):
        super(UNet1DPyTorch, self).__init__()
        self.input_length = input_length
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Encoder
        self.enc1 = self.conv_block(input_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder (with skip connections, so input channels are doubled)
        self.dec4 = self.decoder_block(1024 + 512, 512)  # 1024 + 512 for skip connection
        self.dec3 = self.decoder_block(512 + 256, 256)   # 512 + 256 for skip connection
        self.dec2 = self.decoder_block(256 + 128, 128)   # 256 + 128 for skip connection
        self.dec1 = self.decoder_block(128 + 64, 64)     # 128 + 64 for skip connection
        
        # Final upsampling and output
        self.final_upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.final_conv = nn.Conv1d(64, num_classes, kernel_size=1)
        self.final_activation = nn.Softmax(dim=1)  # Changed to Softmax for multi-class
        
        # Pooling layers
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def conv_block(self, in_channels, out_channels):
        """Convolutional block with batch normalization and ReLU."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def decoder_block(self, in_channels, out_channels):
        """Decoder block with upsampling and convolution."""
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        # Input should already be in correct format (batch_size, channels, length)
        
        # Encoder path
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder path with skip connections
        # Upsample bottleneck to match e4 size
        b_up = F.interpolate(b, size=e4.size(2), mode='linear', align_corners=False)
        d4 = torch.cat([b_up, e4], dim=1)
        d4 = self.dec4(d4)
        
        # Upsample d4 to match e3 size
        d4_up = F.interpolate(d4, size=e3.size(2), mode='linear', align_corners=False)
        d3 = torch.cat([d4_up, e3], dim=1)
        d3 = self.dec3(d3)
        
        # Upsample d3 to match e2 size
        d3_up = F.interpolate(d3, size=e2.size(2), mode='linear', align_corners=False)
        d2 = torch.cat([d3_up, e2], dim=1)
        d2 = self.dec2(d2)
        
        # Upsample d2 to match e1 size
        d2_up = F.interpolate(d2, size=e1.size(2), mode='linear', align_corners=False)
        d1 = torch.cat([d2_up, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Final upsampling and output
        out = self.final_upsample(d1)
        # Ensure output matches input length
        if out.size(2) != x.size(2):
            out = F.interpolate(out, size=x.size(2), mode='linear', align_corners=False)
        out = self.final_conv(out)
        out = self.final_activation(out)
        
        return out
    
    def dice_coefficient(self, y_pred, y_true, smooth=1e-6):
        """Calculate Dice coefficient for multi-class segmentation."""
        # Convert predictions to class indices
        y_pred_classes = torch.argmax(y_pred, dim=1)  # Shape: (batch_size, length)
        y_true_classes = y_true.squeeze(1).long()  # Shape: (batch_size, length)
        
        # Calculate Dice for each class
        dice_scores = []
        for class_id in range(self.num_classes):
            pred_mask = (y_pred_classes == class_id).float()
            true_mask = (y_true_classes == class_id).float()
            
            intersection = (pred_mask * true_mask).sum()
            union = pred_mask.sum() + true_mask.sum()
            
            dice = (2.0 * intersection + smooth) / (union + smooth)
            dice_scores.append(dice)
        
        # Return mean Dice across all classes
        return torch.stack(dice_scores).mean()
    
    def iou_metric(self, y_pred, y_true, smooth=1e-6):
        """Calculate Intersection over Union (IoU) for multi-class segmentation."""
        # Convert predictions to class indices
        y_pred_classes = torch.argmax(y_pred, dim=1)  # Shape: (batch_size, length)
        y_true_classes = y_true.squeeze(1).long()  # Shape: (batch_size, length)
        
        # Calculate IoU for each class
        iou_scores = []
        for class_id in range(self.num_classes):
            pred_mask = (y_pred_classes == class_id).float()
            true_mask = (y_true_classes == class_id).float()
            
            intersection = (pred_mask * true_mask).sum()
            union = pred_mask.sum() + true_mask.sum() - intersection
            
            iou = (intersection + smooth) / (union + smooth)
            iou_scores.append(iou)
        
        # Return mean IoU across all classes
        return torch.stack(iou_scores).mean()

class SignalDataset(Dataset):
    """Dataset class for signal data."""
    
    def __init__(self, X, y):
        # Convert to PyTorch tensors and ensure correct shape
        self.X = torch.FloatTensor(X)  # Shape: (batch_size, length, channels)
        self.y = torch.FloatTensor(y)  # Shape: (batch_size, length, channels)
        
        # Transpose to (batch_size, channels, length) for PyTorch conv1d
        self.X = self.X.transpose(1, 2)  # Shape: (batch_size, channels, length)
        self.y = self.y.transpose(1, 2)  # Shape: (batch_size, channels, length)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class UNetTrainer:
    """Training class for U-Net model."""
    
    def __init__(self, model, device='cpu', learning_rate=0.0001):  # Reduced learning rate
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Weighted loss for imbalanced classes - 10x weight for Class 1 (diastolic CC)
        class_weights = torch.tensor([1.0, 10.0, 3.0, 5.0]).to(device)  # [nothing, diastolic_CC, systolic, diastolic_NH]
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_dice = 0
        total_iou = 0
        
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Convert target to class indices for CrossEntropyLoss
            batch_y_indices = batch_y.squeeze(1).long()  # Shape: (batch_size, length)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)  # Shape: (batch_size, num_classes, length)
            
            # Reshape for CrossEntropyLoss: (batch_size, num_classes, length) -> (batch_size * length, num_classes)
            outputs_flat = outputs.permute(0, 2, 1).contiguous().view(-1, outputs.size(1))
            targets_flat = batch_y_indices.view(-1)
            
            loss = self.criterion(outputs_flat, targets_flat)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                # Convert outputs back to probability format for metrics
                outputs_prob = F.softmax(outputs, dim=1)
                dice = self.model.dice_coefficient(outputs_prob, batch_y)
                iou = self.model.iou_metric(outputs_prob, batch_y)
                
                total_loss += loss.item()
                total_dice += dice.item()
                total_iou += iou.item()
        
        return total_loss / len(dataloader), total_dice / len(dataloader), total_iou / len(dataloader)
    
    def validate(self, dataloader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_dice = 0
        total_iou = 0
        
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Convert target to class indices for CrossEntropyLoss
                batch_y_indices = batch_y.squeeze(1).long()  # Shape: (batch_size, length)
                
                outputs = self.model(batch_X)  # Shape: (batch_size, num_classes, length)
                
                # Reshape for CrossEntropyLoss: (batch_size, num_classes, length) -> (batch_size * length, num_classes)
                outputs_flat = outputs.permute(0, 2, 1).contiguous().view(-1, outputs.size(1))
                targets_flat = batch_y_indices.view(-1)
                
                loss = self.criterion(outputs_flat, targets_flat)
                
                # Convert outputs back to probability format for metrics
                outputs_prob = F.softmax(outputs, dim=1)
                dice = self.model.dice_coefficient(outputs_prob, batch_y)
                iou = self.model.iou_metric(outputs_prob, batch_y)
                
                total_loss += loss.item()
                total_dice += dice.item()
                total_iou += iou.item()
        
        return total_loss / len(dataloader), total_dice / len(dataloader), total_iou / len(dataloader)
    
    def train(self, train_loader, val_loader, epochs=100, patience=20):
        """Train the model with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        train_dices = []
        val_dices = []
        train_ious = []
        val_ious = []
        
        for epoch in range(epochs):
            # Training
            train_loss, train_dice, train_iou = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_dice, val_iou = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_dices.append(train_dice)
            val_dices.append(val_dice)
            train_ious.append(train_iou)
            val_ious.append(val_iou)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Train IoU: {train_iou:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_dices': train_dices,
            'val_dices': val_dices,
            'train_ious': train_ious,
            'val_ious': val_ious
        }
    
    def predict(self, dataloader):
        """Make predictions."""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def evaluate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics for multi-class segmentation."""
        # Convert predictions to class indices
        y_pred_classes = np.argmax(y_pred, axis=1)  # Shape: (batch_size, length)
        # Handle both (batch_size, length, 1) and (batch_size, length) shapes
        if y_true.ndim == 3 and y_true.shape[2] == 1:
            y_true_classes = y_true.squeeze(2)  # Shape: (batch_size, length)
        else:
            y_true_classes = y_true  # Already (batch_size, length)
        
        # Flatten for metric calculation
        y_true_flat = y_true_classes.flatten()
        y_pred_flat = y_pred_classes.flatten()
        
        # Calculate overall accuracy
        accuracy = np.mean(y_true_flat == y_pred_flat)
        
        # Get number of classes from the model
        num_classes = self.model.num_classes
        
        # Calculate per-class metrics
        precision_scores = []
        recall_scores = []
        f1_scores = []
        dice_scores = []
        iou_scores = []
        
        for class_id in range(num_classes):
            # Binary masks for this class
            y_true_binary = (y_true_flat == class_id).astype(np.float32)
            y_pred_binary = (y_pred_flat == class_id).astype(np.float32)
            
            # Precision, Recall, F1
            if np.sum(y_true_binary) > 0 or np.sum(y_pred_binary) > 0:
                precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            else:
                precision = recall = f1 = 0.0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            
            # Dice and IoU
            intersection = np.sum(y_true_binary * y_pred_binary)
            union = np.sum(y_true_binary) + np.sum(y_pred_binary)
            dice = (2.0 * intersection) / (union + 1e-6) if union > 0 else 0.0
            dice_scores.append(dice)
            
            union_iou = np.sum(y_true_binary) + np.sum(y_pred_binary) - intersection
            iou = intersection / (union_iou + 1e-6) if union_iou > 0 else 0.0
            iou_scores.append(iou)
        
        # Calculate macro averages
        macro_precision = np.mean(precision_scores)
        macro_recall = np.mean(recall_scores)
        macro_f1 = np.mean(f1_scores)
        macro_dice = np.mean(dice_scores)
        macro_iou = np.mean(iou_scores)
        
        return {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'macro_dice': macro_dice,
            'macro_iou': macro_iou,
            'per_class_precision': precision_scores,
            'per_class_recall': recall_scores,
            'per_class_f1': f1_scores,
            'per_class_dice': dice_scores,
            'per_class_iou': iou_scores
        }

def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    """Create data loaders for training and validation."""
    train_dataset = SignalDataset(X_train, y_train)
    val_dataset = SignalDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test the model
    print("Testing PyTorch U-Net model...")
    
    # Create dummy data
    X_test = np.random.randn(10, 400, 1)
    y_test = np.random.randint(0, 2, (10, 400, 1)).astype(np.float32)
    
    # Create model
    model = UNet1DPyTorch(input_length=400, input_channels=1, num_classes=1)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    X_tensor = torch.FloatTensor(X_test)
    with torch.no_grad():
        output = model(X_tensor)
    
    print(f"Input shape: {X_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print("PyTorch U-Net model test completed successfully!")
