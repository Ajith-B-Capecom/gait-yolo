"""
Compare YOLO vs Background Subtraction Methods
Test script to compare silhouette extraction quality and speed
"""

import cv2
import time
import numpy as np
from pathlib import Path
from scripts.silhouette_extraction import SilhouetteExtractor


def compare_methods(image_path, output_dir='comparison_output'):
    """
    Compare different silhouette extraction methods on a single image
    
    Args:
        image_path (str): Path to test image
        output_dir (str): Directory to save comparison results
    """
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Cannot read image {image_path}")
        return
    
    print(f"\nComparing silhouette extraction methods")
    print(f"Image: {image_path}")
    print(f"Size: {frame.shape[1]}x{frame.shape[0]}")
    print("="*60)
    
    methods = [
        ('YOLO (nano)', 'yolo', 'yolo11n-seg.pt'),
        ('YOLO (small)', 'yolo', 'yolo11s-seg.pt'),
        ('MOG2', 'mog2', None),
        ('KNN', 'knn', None),
    ]
    
    results = []
    
    for method_name, method, model in methods:
        print(f"\n{method_name}:")
        print("-" * 40)
        
        try:
            # Initialize extractor
            if method == 'yolo':
                extractor = SilhouetteExtractor(method=method, model=model, conf=0.25, device='cpu')
            else:
                extractor = SilhouetteExtractor(method=method)
            
            # Measure time
            start_time = time.time()
            
            # Extract silhouette
            if method in ['mog2', 'knn']:
                # Background subtraction needs multiple frames
                # Apply frame multiple times to build background model
                for _ in range(10):
                    extractor.get_silhouette(frame)
            
            silhouette = extractor.get_silhouette(frame)
            
            elapsed_time = time.time() - start_time
            
            # Save result
            output_path = f"{output_dir}/{method_name.replace(' ', '_').lower()}.jpg"
            cv2.imwrite(output_path, silhouette)
            
            # Calculate metrics
            white_pixels = np.sum(silhouette > 127)
            total_pixels = silhouette.shape[0] * silhouette.shape[1]
            coverage = (white_pixels / total_pixels) * 100
            
            print(f"  ✓ Time: {elapsed_time:.3f} seconds")
            print(f"  ✓ Coverage: {coverage:.2f}%")
            print(f"  ✓ Saved: {output_path}")
            
            results.append({
                'method': method_name,
                'time': elapsed_time,
                'coverage': coverage,
                'path': output_path
            })
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    # Create comparison image
    print("\n" + "="*60)
    print("Creating comparison visualization...")
    
    comparison_images = []
    labels = []
    
    # Add original image
    comparison_images.append(frame)
    labels.append("Original")
    
    # Add silhouettes
    for result in results:
        silhouette = cv2.imread(result['path'], cv2.IMREAD_GRAYSCALE)
        if silhouette is not None:
            # Convert to BGR for concatenation
            silhouette_bgr = cv2.cvtColor(silhouette, cv2.COLOR_GRAY2BGR)
            comparison_images.append(silhouette_bgr)
            labels.append(f"{result['method']}\n{result['time']:.2f}s")
    
    # Resize all images to same height
    target_height = 400
    resized_images = []
    for img in comparison_images:
        h, w = img.shape[:2]
        new_w = int(w * target_height / h)
        resized = cv2.resize(img, (new_w, target_height))
        resized_images.append(resized)
    
    # Add labels to images
    labeled_images = []
    for img, label in zip(resized_images, labels):
        # Create copy
        img_copy = img.copy()
        
        # Add white background for text
        cv2.rectangle(img_copy, (0, 0), (img.shape[1], 60), (255, 255, 255), -1)
        
        # Add text
        lines = label.split('\n')
        for i, line in enumerate(lines):
            cv2.putText(img_copy, line, (10, 25 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        labeled_images.append(img_copy)
    
    # Concatenate horizontally
    comparison = np.hstack(labeled_images)
    
    # Save comparison
    comparison_path = f"{output_dir}/comparison.jpg"
    cv2.imwrite(comparison_path, comparison)
    print(f"✓ Comparison saved: {comparison_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Method':<20} {'Time (s)':<12} {'Coverage (%)':<15}")
    print("-" * 60)
    for result in results:
        print(f"{result['method']:<20} {result['time']:<12.3f} {result['coverage']:<15.2f}")
    
    # Find fastest and best coverage
    fastest = min(results, key=lambda x: x['time'])
    best_coverage = max(results, key=lambda x: x['coverage'])
    
    print("\n" + "="*60)
    print(f"⚡ Fastest: {fastest['method']} ({fastest['time']:.3f}s)")
    print(f"🎯 Best Coverage: {best_coverage['method']} ({best_coverage['coverage']:.2f}%)")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Try to find a test image
        test_paths = [
            'data/frames/person1/*/frame_000000.jpg',
            'data/frames/*/frame_000000.jpg',
            'test_image.jpg',
            'frame.jpg'
        ]
        
        image_path = None
        for pattern in test_paths:
            matches = list(Path('.').glob(pattern))
            if matches:
                image_path = str(matches[0])
                break
        
        if image_path is None:
            print("Usage: python compare_methods.py <image_path>")
            print("\nOr place a test image in one of these locations:")
            for path in test_paths:
                print(f"  - {path}")
            sys.exit(1)
    
    compare_methods(image_path)
