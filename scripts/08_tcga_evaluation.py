#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenSlideFM - TCGA Evaluation Pipeline
============================================================

This script is part of the OpenSlideFM pipeline for computational pathology.

Paper: "OpenSlideFM: A Resource-Efficient Foundation Model for 
        Computational Pathology on Whole Slide Images"

Authors: Sanwal Ahmad Zafar, Wei Qin
Institution: Shanghai Jiao Tong University

License: Apache 2.0
"""

# COMPLETE TCGA EVALUATION PIPELINE
# Trains and evaluates cancer classification on TCGA dataset to establish baseline performance for comparison with external validation (CAM16/17/PANDA).

import os
import sys
import json
import warnings
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    classification_report, confusion_matrix,
    balanced_accuracy_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Paths
    OPENSLIDE = Path(os.environ.get("WORKSPACE", "./workspace")))
    DL_V2 = Path(os.environ.get("DATA_ROOT", "./data")))
    
    # Data
    EMBEDDINGS = DL_V2 / "artifacts" / "embeddings" / "patient_means_clean_run_20250908_020405_emb_openclip_vitb16_turbo.parquet"
    LABELS = DL_V2 / "artifacts" / "labels" / "labels.csv"
    MANIFEST = OPENSLIDE / "manifests" / "manifest_tcga.csv"
    
    # Output
    OUTPUT = OPENSLIDE / "results" / "tcga_baseline_evaluation"
    
    # Model
    HIDDEN_DIM = 256
    DROPOUT = 0.3
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
    BATCH_SIZE = 64
    PATIENCE = 10
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = Config()

# ============================================================================
# UTILITIES
# ============================================================================
def print_header(text):
    print(f"\n{'='*80}")
    print(f" {text}")
    print('='*80)

def print_subheader(text):
    print(f"\n{'-'*80}")
    print(f" {text}")
    print('-'*80)

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    """Load embeddings, labels, and manifest"""
    print_header("1. LOADING DATA")
    
    # Load embeddings
    print("\n📦 Loading embeddings...")
    df_emb = pd.read_parquet(CFG.EMBEDDINGS)
    print(f"  ✓ Embeddings: {df_emb.shape}")
    print(f"    Patients: {len(df_emb)}")
    print(f"    Features: {df_emb.shape[1]}")
    
    # Load labels
    print("\n📋 Loading labels...")
    df_labels = pd.read_csv(CFG.LABELS)
    print(f"  ✓ Labels: {df_labels.shape}")
    
    # Check split distribution
    if 'split' in df_labels.columns:
        split_dist = df_labels['split'].value_counts()
        print(f"\n  Split distribution:")
        for split, count in split_dist.items():
            print(f"    {split}: {count}")
    
    # Load manifest for cancer codes
    print("\n🗂️  Loading manifest...")
    df_manifest = pd.read_csv(CFG.MANIFEST)
    print(f"  ✓ Manifest: {df_manifest.shape}")
    print(f"    Total slides: {len(df_manifest)}")
    print(f"    Cancer types: {df_manifest['cancer_code'].nunique()}")
    
    return df_emb, df_labels, df_manifest

def prepare_dataset(df_emb, df_labels, df_manifest):
    """Prepare train/test datasets with labels"""
    print_header("2. PREPARING DATASET")
    
    # Extract patient IDs from embeddings index
    print("\n🔗 Mapping patients to cancer types...")
    
    # Get patient-to-cancer mapping from manifest
    # Extract patient ID from slide_id (e.g., TCGA-02-0001-01A-01-TS1 -> TCGA-02-0001)
    df_manifest['patient_id'] = df_manifest['slide_id'].str.extract(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', expand=False)
    
    # Get unique patient-cancer mapping
    patient_cancer_map = df_manifest.groupby('patient_id')['cancer_code'].first().to_dict()
    
    # Map embeddings to cancer types
    df_emb['patient_id'] = df_emb.index
    df_emb['cancer_type'] = df_emb['patient_id'].map(patient_cancer_map)
    
    # Remove patients without cancer labels
    df_emb_labeled = df_emb[df_emb['cancer_type'].notna()].copy()
    print(f"  ✓ Patients with labels: {len(df_emb_labeled)}")
    print(f"    Removed {len(df_emb) - len(df_emb_labeled)} patients without labels")
    
    # Add split information from labels.csv if available
    if 'split' in df_labels.columns:
        # Create patient-split mapping
        df_labels['patient_id'] = df_labels['patient']
        patient_split_map = df_labels.set_index('patient_id')['split'].to_dict()
        df_emb_labeled['split'] = df_emb_labeled['patient_id'].map(patient_split_map)
        
        # Use patients with defined splits
        df_emb_labeled = df_emb_labeled[df_emb_labeled['split'].notna()].copy()
        print(f"  ✓ Patients with train/test split: {len(df_emb_labeled)}")
    else:
        # Create random split if none exists
        print("  ⚠️  No split found, creating 80/10/10 split...")
        from sklearn.model_selection import train_test_split
        patients = df_emb_labeled['patient_id'].values
        train_val, test = train_test_split(patients, test_size=0.1, random_state=42)
        train, val = train_test_split(train_val, test_size=0.111, random_state=42)  # 0.111 * 0.9 ≈ 0.1
        
        split_map = {}
        for p in train: split_map[p] = 'train'
        for p in val: split_map[p] = 'val'
        for p in test: split_map[p] = 'test'
        df_emb_labeled['split'] = df_emb_labeled['patient_id'].map(split_map)
    
    # Show cancer type distribution
    print(f"\n📊 Cancer type distribution:")
    cancer_counts = df_emb_labeled['cancer_type'].value_counts()
    print(f"  Total cancer types: {len(cancer_counts)}")
    print(f"  Top 10:")
    for cancer, count in cancer_counts.head(10).items():
        print(f"    {cancer}: {count}")
    
    # Show split distribution
    print(f"\n📊 Split distribution:")
    for split in ['train', 'val', 'test']:
        count = (df_emb_labeled['split'] == split).sum()
        print(f"  {split}: {count}")
    
    # Prepare feature matrix and labels
    feature_cols = [c for c in df_emb_labeled.columns if c.startswith('f')]
    X = df_emb_labeled[feature_cols].values.astype(np.float32)
    
    # Encode cancer types
    le = LabelEncoder()
    y = le.fit_transform(df_emb_labeled['cancer_type'].values)
    
    print(f"\n✓ Feature matrix: {X.shape}")
    print(f"✓ Number of classes: {len(le.classes_)}")
    
    # Split data
    train_mask = df_emb_labeled['split'] == 'train'
    val_mask = df_emb_labeled['split'] == 'val'
    test_mask = df_emb_labeled['split'] == 'test'
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\n✓ Train: {X_train.shape}, {len(np.unique(y_train))} classes")
    print(f"✓ Val:   {X_val.shape}, {len(np.unique(y_val))} classes")
    print(f"✓ Test:  {X_test.shape}, {len(np.unique(y_test))} classes")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), le, df_emb_labeled

# ============================================================================
# MODEL
# ============================================================================
class CancerClassifier(nn.Module):
    """Simple MLP for cancer classification"""
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# ============================================================================
# TRAINING
# ============================================================================
def train_model(train_data, val_data, num_classes):
    """Train cancer classifier"""
    print_header("3. TRAINING MODEL")
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).long()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val).long()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = CancerClassifier(
        input_dim=input_dim,
        hidden_dim=CFG.HIDDEN_DIM,
        num_classes=num_classes,
        dropout=CFG.DROPOUT
    ).to(CFG.DEVICE)
    
    print(f"\n🧠 Model architecture:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {CFG.HIDDEN_DIM}")
    print(f"  Output classes: {num_classes}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.LEARNING_RATE,
        weight_decay=CFG.WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print(f"\n🚀 Starting training...")
    print(f"  Epochs: {CFG.EPOCHS}")
    print(f"  Batch size: {CFG.BATCH_SIZE}")
    print(f"  Learning rate: {CFG.LEARNING_RATE}")
    print(f"  Device: {CFG.DEVICE}")
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(1, CFG.EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(CFG.DEVICE)
            y_batch = y_batch.to(CFG.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(CFG.DEVICE)
                y_batch = y_batch.to(CFG.DEVICE)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_true, val_preds)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{CFG.EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            
            # Ensure output directory exists
            CFG.OUTPUT.mkdir(parents=True, exist_ok=True)
            
            # Save best model (workaround for Unicode path issue)
            # Save to temp file, then copy using pure Python binary I/O
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pth') as tmp:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, tmp)
                tmp_path = tmp.name
            
            # Copy using pure Python binary I/O (handles Unicode)
            final_path = CFG.OUTPUT / "best_model.pth"
            try:
                with open(tmp_path, 'rb') as src:
                    with open(final_path, 'wb') as dst:
                        dst.write(src.read())
            finally:
                os.unlink(tmp_path)  # Delete temp file
        else:
            patience_counter += 1
            if patience_counter >= CFG.PATIENCE:
                print(f"\n⏸️  Early stopping triggered at epoch {epoch}")
                break
    
    print(f"\n✓ Training complete!")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    
    # Load best model (workaround for Unicode path)
    model_path = str(CFG.OUTPUT / "best_model.pth")
    with open(model_path, 'rb') as f:
        checkpoint = torch.load(f, map_location=CFG.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history

# ============================================================================
# EVALUATION
# ============================================================================
@torch.no_grad()
def evaluate_model(model, test_data, label_encoder):
    """Evaluate on test set"""
    print_header("4. EVALUATING ON TEST SET")
    
    X_test, y_test = test_data
    
    # Create dataloader
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test).long()
    )
    test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False)
    
    # Predict
    model.eval()
    all_preds = []
    all_probs = []
    all_true = []
    
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(CFG.DEVICE)
        outputs = model(X_batch)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_true.extend(y_batch.numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.vstack(all_probs)
    all_true = np.array(all_true)
    
    # Calculate metrics
    print("\n📊 Test Set Performance:")
    
    # Overall accuracy
    acc = accuracy_score(all_true, all_preds)
    print(f"\n  Overall Accuracy: {acc:.4f}")
    
    # Balanced accuracy
    bal_acc = balanced_accuracy_score(all_true, all_preds)
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    
    # Macro F1
    macro_f1 = f1_score(all_true, all_preds, average='macro')
    print(f"  Macro F1: {macro_f1:.4f}")
    
    # Weighted F1
    weighted_f1 = f1_score(all_true, all_preds, average='weighted')
    print(f"  Weighted F1: {weighted_f1:.4f}")
    
    # Multi-class AUC (one-vs-rest)
    try:
        auc_ovr = roc_auc_score(all_true, all_probs, multi_class='ovr', average='macro')
        print(f"  Macro AUC (OvR): {auc_ovr:.4f}")
    except:
        auc_ovr = None
        print(f"  Macro AUC (OvR): N/A")
    
    # Per-class metrics
    print(f"\n📋 Classification Report:")
    class_names = label_encoder.classes_
    report = classification_report(
        all_true, all_preds,
        target_names=class_names,
        digits=3
    )
    print(report)
    
    # Save detailed metrics
    results = {
        'overall': {
            'accuracy': float(acc),
            'balanced_accuracy': float(bal_acc),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'macro_auc_ovr': float(auc_ovr) if auc_ovr is not None else None,
            'num_samples': int(len(all_true)),
            'num_classes': int(len(class_names))
        },
        'per_class': classification_report(
            all_true, all_preds,
            target_names=class_names,
            output_dict=True
        )
    }
    
    # Confusion matrix
    cm = confusion_matrix(all_true, all_preds)
    
    return results, cm, all_preds, all_probs, all_true

# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_results(history, cm, label_encoder):
    """Create visualization plots"""
    print_header("5. CREATING VISUALIZATIONS")
    
    fig = plt.figure(figsize=(20, 5))
    
    # Training curves
    ax1 = plt.subplot(1, 3, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Validation accuracy
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(epochs, history['val_acc'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Confusion matrix (top 15 classes by support)
    ax3 = plt.subplot(1, 3, 3)
    
    # Get top classes
    class_support = cm.sum(axis=1)
    top_indices = np.argsort(class_support)[-15:][::-1]
    cm_top = cm[np.ix_(top_indices, top_indices)]
    class_names_top = label_encoder.classes_[top_indices]
    
    sns.heatmap(cm_top, annot=False, fmt='d', cmap='Blues', 
                xticklabels=class_names_top, yticklabels=class_names_top,
                ax=ax3, cbar_kws={'label': 'Count'})
    ax3.set_title('Confusion Matrix (Top 15 Classes)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Predicted', fontsize=12)
    ax3.set_ylabel('True', fontsize=12)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax3.get_yticklabels(), rotation=0, fontsize=8)
    
    plt.tight_layout()
    plot_path = str(CFG.OUTPUT / 'training_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: training_results.png")
    plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================
def save_results(results, cm, label_encoder, df_labeled, y_pred, y_true):
    """Save all results to disk"""
    print_header("6. SAVING RESULTS")
    
    # Save metrics JSON
    with open(str(CFG.OUTPUT / 'test_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved: test_metrics.json")
    
    # Save confusion matrix
    cm_df = pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    )
    cm_df.to_csv(str(CFG.OUTPUT / 'confusion_matrix.csv'))
    print(f"  ✓ Saved: confusion_matrix.csv")
    
    # Save per-class metrics
    per_class_df = pd.DataFrame(results['per_class']).T
    per_class_df.to_csv(str(CFG.OUTPUT / 'per_class_metrics.csv'))
    print(f"  ✓ Saved: per_class_metrics.csv")
    
    # Save predictions
    test_mask = df_labeled['split'] == 'test'
    test_patients = df_labeled[test_mask]['patient_id'].values
    
    pred_df = pd.DataFrame({
        'patient_id': test_patients,
        'true_label': label_encoder.inverse_transform(y_true),
        'pred_label': label_encoder.inverse_transform(y_pred),
        'correct': y_true == y_pred
    })
    pred_df.to_csv(str(CFG.OUTPUT / 'test_predictions.csv'), index=False)
    print(f"  ✓ Saved: test_predictions.csv")
    
    # Create summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'TCGA',
        'num_patients': len(df_labeled),
        'num_classes': len(label_encoder.classes_),
        'train_size': int((df_labeled['split'] == 'train').sum()),
        'val_size': int((df_labeled['split'] == 'val').sum()),
        'test_size': int((df_labeled['split'] == 'test').sum()),
        'model': {
            'type': 'MLP',
            'hidden_dim': CFG.HIDDEN_DIM,
            'dropout': CFG.DROPOUT,
            'learning_rate': CFG.LEARNING_RATE,
            'weight_decay': CFG.WEIGHT_DECAY
        },
        'results': results['overall']
    }
    
    with open(str(CFG.OUTPUT / 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved: summary.json")
    
    print(f"\n✓ All results saved to: {CFG.OUTPUT}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    """Main execution"""
    print("="*80)
    print(" TCGA BASELINE EVALUATION PIPELINE")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {CFG.OUTPUT}")
    
    # Ensure output directory exists
    CFG.OUTPUT.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    df_emb, df_labels, df_manifest = load_data()
    
    # 2. Prepare dataset
    train_data, val_data, test_data, label_encoder, df_labeled = prepare_dataset(
        df_emb, df_labels, df_manifest
    )
    
    # 3. Train model
    model, history = train_model(train_data, val_data, num_classes=len(label_encoder.classes_))
    
    # 4. Evaluate
    results, cm, y_pred, y_probs, y_true = evaluate_model(model, test_data, label_encoder)
    
    # 5. Visualize
    plot_results(history, cm, label_encoder)
    
    # 6. Save
    save_results(results, cm, label_encoder, df_labeled, y_pred, y_true)
    
    # Final summary
    print_header("SUMMARY")
    print(f"\n✅ TCGA Baseline Evaluation Complete!")
    print(f"\n📊 Key Metrics:")
    print(f"  Test Accuracy: {results['overall']['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {results['overall']['balanced_accuracy']:.4f}")
    print(f"  Macro F1: {results['overall']['macro_f1']:.4f}")
    if results['overall']['macro_auc_ovr'] is not None:
        print(f"  Macro AUC (OvR): {results['overall']['macro_auc_ovr']:.4f}")
    
    print(f"\n📁 Results saved to: {CFG.OUTPUT}")
    print(f"\n💡 Next Steps:")
    print(f"  1. Compare TCGA test metrics with CAM16/17/PANDA")
    print(f"  2. Calculate performance drop: (CAM16 - TCGA test)")
    print(f"  3. Include in publication tables")
    
    print("\n" + "="*80)
    print(" COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()