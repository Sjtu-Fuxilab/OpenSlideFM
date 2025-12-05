#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenSlideFM - PANDA Processing Pipeline
============================================================

This script is part of the OpenSlideFM pipeline for computational pathology.

Paper: "OpenSlideFM: A Resource-Efficient Foundation Model for 
        Computational Pathology on Whole Slide Images"

Authors: Sanwal Ahmad Zafar, Wei Qin
Institution: Shanghai Jiao Tong University

License: Apache 2.0
"""

# PANDA Processing Pipeline

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings('ignore')

# Configuration
PANDA_ROOT = Path(r"D:\个人文件夹\Sanwal\OpenSlide\Validation Data\PANDA")
WORKSPACE = Path(r"D:\个人文件夹\Sanwal\OpenSlide")

OUTPUT_DIRS = {
    "features_05": WORKSPACE / "features" / "panda" / "scale0p5",
    "features_20": WORKSPACE / "features" / "panda" / "scale2p0", 
    "results": WORKSPACE / "results" / "panda",
    "logs": WORKSPACE / "logs" / "panda"
}
for d in OUTPUT_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# Optimization settings
N_WORKERS = min(cpu_count() - 1, 8)  # Leave one CPU free
BATCH_SIZE = 128  # Increased batch size
PREFETCH_TILES = 4  # Prefetch multiple tiles
USE_MIXED_PRECISION = True
CACHE_SIZE = 1000  # Cache recent tiles in memory

print(f"System info: {cpu_count()} CPUs available, using {N_WORKERS} workers")

def check_already_processed(image_id):
    """Quick check if slide is already processed"""
    feat_05 = OUTPUT_DIRS["features_05"] / f"{image_id}.npy"
    feat_20 = OUTPUT_DIRS["features_20"] / f"{image_id}.npy"
    
    if feat_05.exists() and feat_20.exists():
        # Verify files are valid
        try:
            f05 = np.load(feat_05, mmap_mode='r')
            f20 = np.load(feat_20, mmap_mode='r')
            if f05.shape[1] == 768 and f20.shape[1] == 768:
                return True
        except:
            # Corrupted files, will reprocess
            pass
    return False

def get_pending_slides(df, max_slides=None):
    """Get list of slides that need processing"""
    pending = []
    
    for idx, row in df.iterrows():
        if not row['image_exists']:
            continue
            
        image_id = row['image_id']
        
        # Skip if already processed
        if check_already_processed(image_id):
            continue
        
        pending.append(row)
        
        if max_slides and len(pending) >= max_slides:
            break
    
    return pending

def process_single_slide(args):
    """Process a single slide - can be run in parallel"""
    row, device_id = args
    
    # Import heavy libraries only in worker process
    import torch
    import torchvision.models as tvm
    import torch.nn as nn
    from PIL import Image
    import openslide
    
    # Set device for this worker
    if torch.cuda.is_available():
        device = f"cuda:{device_id % torch.cuda.device_count()}"
    else:
        device = "cpu"
    
    image_id = row['image_id']
    image_path = Path(row['image_path'])
    
    # Double-check if already processed
    if check_already_processed(image_id):
        return image_id, "skipped", 0
    
    # Build model
    class ConvNeXtTinyFeats(nn.Module):
        def __init__(self):
            super().__init__()
            weights = tvm.ConvNeXt_Tiny_Weights.DEFAULT
            model = tvm.convnext_tiny(weights=weights)
            self.features = model.features
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.eval()
            for p in self.parameters(): 
                p.requires_grad = False
        
        @torch.no_grad()
        def forward(self, x):
            x = self.features(x)
            x = self.gap(x).flatten(1)
            return x
    
    try:
        model = ConvNeXtTinyFeats().to(device)
        if device != "cpu":
            model = model.to(memory_format=torch.channels_last)
        
        # Open slide
        slide = openslide.OpenSlide(str(image_path))
        
        # Configuration
        TILE_SIZE = 256
        STRIDE = 224
        SCALES = [0.5, 2.0]
        MAX_TILES = {0.5: 1200, 2.0: 400}
        
        tiles_extracted = 0
        
        # Process each scale
        for scale in SCALES:
            scale_dir = OUTPUT_DIRS[f"features_{str(scale).replace('.','').replace('p','')}"]
            feat_path = scale_dir / f"{image_id}.npy"
            
            if feat_path.exists():
                continue
            
            # Determine level
            base_mpp = 0.5
            target_downsample = scale / base_mpp
            level = slide.get_best_level_for_downsample(target_downsample)
            actual_downsample = slide.level_downsamples[level]
            
            # Get dimensions
            level_w, level_h = slide.level_dimensions[level]
            
            # Collect tiles efficiently
            tiles = []
            tile_batch = []
            
            for y in range(0, level_h - TILE_SIZE + 1, STRIDE):
                for x in range(0, level_w - TILE_SIZE + 1, STRIDE):
                    if len(tiles) >= MAX_TILES[scale]:
                        break
                    
                    # Read tile
                    x0 = int(x * actual_downsample)
                    y0 = int(y * actual_downsample)
                    tile = slide.read_region((x0, y0), level, (TILE_SIZE, TILE_SIZE)).convert('RGB')
                    
                    # Quick tissue check
                    tile_np = np.array(tile)
                    if tile_np.mean() < 235 and tile_np.std() > 15:
                        # Resize immediately
                        tile_224 = tile.resize((224, 224), Image.BILINEAR)
                        tile_batch.append(tile_224)
                        
                        # Process batch when full
                        if len(tile_batch) >= BATCH_SIZE:
                            batch_features = process_batch(tile_batch, model, device)
                            tiles.extend(batch_features)
                            tile_batch = []
                            tiles_extracted += len(batch_features)
                
                if len(tiles) >= MAX_TILES[scale]:
                    break
            
            # Process remaining tiles
            if tile_batch:
                batch_features = process_batch(tile_batch, model, device)
                tiles.extend(batch_features)
                tiles_extracted += len(batch_features)
            
            # Save features
            if tiles:
                all_features = np.vstack(tiles).astype(np.float16)
                np.save(feat_path, all_features)
            else:
                np.save(feat_path, np.zeros((0, 768), dtype=np.float16))
        
        slide.close()
        
        # Clean up GPU memory
        if device != "cpu":
            torch.cuda.empty_cache()
        
        return image_id, "success", tiles_extracted
        
    except Exception as e:
        return image_id, f"error: {str(e)}", 0

def process_batch(tile_batch, model, device):
    """Process a batch of tiles through the model"""
    import torch
    
    # Convert tiles to tensors
    tensors = []
    for tile in tile_batch:
        tile_array = np.array(tile).astype(np.float32) / 255.0
        tensor = torch.from_numpy(tile_array).permute(2, 0, 1)
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        tensors.append(tensor)
    
    # Batch inference
    batch_tensor = torch.stack(tensors).to(device, non_blocking=True)
    if device != "cpu":
        batch_tensor = batch_tensor.to(memory_format=torch.channels_last)
    
    with torch.no_grad():
        if USE_MIXED_PRECISION and device != "cpu":
            with torch.cuda.amp.autocast():
                features = model(batch_tensor)
        else:
            features = model(batch_tensor)
        
        features = features.cpu().numpy()
    
    return features

def extract_features_parallel(df, max_slides=None):
    """Extract features using multiple workers"""
    # Import torch here just for CUDA check
    import torch
    
    print("\n" + "="*80)
    print("PARALLEL FEATURE EXTRACTION")
    print("="*80)
    
    # Get pending slides
    pending_slides = get_pending_slides(df[df['image_exists']], max_slides)
    
    if not pending_slides:
        print("All slides already processed!")
        return
    
    print(f"Found {len(pending_slides)} slides to process")
    print(f"Using {N_WORKERS} parallel workers")
    
    # Check CUDA availability once
    cuda_available = torch.cuda.is_available()
    
    # Prepare arguments for workers
    worker_args = []
    for i, row in enumerate(pending_slides):
        device_id = i % N_WORKERS if cuda_available else 0
        worker_args.append((row, device_id))
    
    # Process in parallel
    results = []
    failed = []
    successful = 0
    skipped = 0
    
    start_time = time.time()
    last_print = start_time
    
    # Use ThreadPoolExecutor for I/O-bound parts, ProcessPoolExecutor for CPU-bound
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_single_slide, args): args[0]['image_id'] 
                  for args in worker_args}
        
        for future in as_completed(futures):
            image_id = futures[future]
            try:
                slide_id, status, tiles = future.result()
                
                if status == "success":
                    successful += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed.append((slide_id, status))
                
                # Progress update
                current_time = time.time()
                if current_time - last_print > 5:  # Print every 5 seconds
                    elapsed = current_time - start_time
                    processed = successful + skipped + len(failed)
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = (len(pending_slides) - processed) / rate if rate > 0 else 0
                    
                    print(f"Progress: {processed}/{len(pending_slides)} | "
                          f"Rate: {rate:.2f} slides/sec | "
                          f"ETA: {eta/60:.1f} min")
                    last_print = current_time
                    
            except Exception as e:
                failed.append((image_id, str(e)))
    
    # Final stats
    elapsed = time.time() - start_time
    print(f"\n" + "="*60)
    print(f"Extraction complete in {elapsed/60:.1f} minutes")
    print(f"Successful: {successful}")
    print(f"Skipped (already done): {skipped}")
    print(f"Failed: {len(failed)}")
    print(f"Average: {successful/elapsed:.2f} slides/sec")
    
    if failed:
        failed_df = pd.DataFrame(failed, columns=['image_id', 'error'])
        failed_df.to_csv(OUTPUT_DIRS["logs"] / "failed_extractions.csv", index=False)

def main_optimized():
    """Optimized main pipeline"""
    import torch
    
    print("="*80)
    print("OPTIMIZED PANDA PROCESSING PIPELINE")
    print(f"Workers: {N_WORKERS} | Batch size: {BATCH_SIZE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print("="*80)
    
    # Load manifest
    manifest_path = OUTPUT_DIRS["logs"] / "panda_manifest.csv"
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
    else:
        # Create manifest
        train_csv = PANDA_ROOT / "train.csv"
        if not train_csv.exists():
            print("ERROR: train.csv not found!")
            return
        
        df = pd.read_csv(train_csv)
        df['image_path'] = df['image_id'].apply(
            lambda x: str(PANDA_ROOT / "train_images" / f"{x}.tiff")
        )
        df['image_exists'] = df['image_path'].apply(lambda x: Path(x).exists())
        df.to_csv(manifest_path, index=False)
    
    print(f"Total slides in dataset: {len(df)}")
    print(f"Slides with images: {df['image_exists'].sum()}")
    
    # Check already processed
    already_done = sum(1 for _, row in df.iterrows() 
                      if row['image_exists'] and check_already_processed(row['image_id']))
    print(f"Already processed: {already_done}")
    
    # Options
    print("\n" + "="*60)
    print("EXTRACTION OPTIONS:")
    print("="*60)
    print(f"1. Quick test (10 slides)")
    print(f"2. Small batch (100 slides)")
    print(f"3. Medium batch (1000 slides)")
    print(f"4. Large batch (5000 slides)")
    print(f"5. Full dataset (all {df['image_exists'].sum()} slides)")
    print(f"6. Skip extraction")
    
    choice = input("\nChoice (1-6): ").strip()
    
    if choice == "6":
        print("Skipping extraction")
        return
    
    max_slides_map = {
        "1": 10,
        "2": 100,
        "3": 1000,
        "4": 5000,
        "5": None
    }
    max_slides = max_slides_map.get(choice, 100)
    
    # Run extraction
    extract_features_parallel(df, max_slides)
    
    print("\nDone! Features saved to:")
    print(f"  {OUTPUT_DIRS['features_05']}")
    print(f"  {OUTPUT_DIRS['features_20']}")

if __name__ == "__main__":
    main_optimized()