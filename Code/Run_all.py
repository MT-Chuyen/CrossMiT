import subprocess
import sys
import os
import re
from pathlib import Path
import numpy as np

def run_fold(fold_num):
    """Run training for a specific fold"""
    print(f"\n{'='*60}")
    print(f"Running Fold {fold_num}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, 'Main.py',
        '--fold', str(fold_num)
    ]
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0

def parse_metrics(result_text):
    """Parse metrics from result text
    Expected format: Best Iter=[idx]@[time] hit=[...], ndcg=[...]
    """
    try:
        # Extract hit and ndcg arrays
        hit_match = re.search(r"hit=\[([\d.,\s]+)\]", result_text)
        ndcg_match = re.search(r"ndcg=\[([\d.,\s]+)\]", result_text)
        
        if hit_match and ndcg_match:
            hits = [float(x.strip().strip("'")) for x in hit_match.group(1).split(',')]
            ndcgs = [float(x.strip().strip("'")) for x in ndcg_match.group(1).split(',')]
            return {'hits': hits, 'ndcgs': ndcgs}
    except Exception as e:
        print(f"Error parsing metrics: {e}")
    
    return None

def extract_fold_results(fold_num):
    """Extract results from fold-specific result file"""
    fold_result_file = f'./output/folds/Fold_{fold_num}_result.txt'
    
    try:
        if os.path.exists(fold_result_file):
            with open(fold_result_file, 'r') as f:
                content = f.read()
                
                # Extract both source and target results
                results = {}
                for domain in ['source result', 'target result']:
                    idx = content.find(domain)
                    if idx != -1:
                        # Get the line after the domain name
                        start = idx + len(domain)
                        end = content.find('\n', start + 1)
                        if end != -1:
                            result_line = content[start:end]
                            metrics = parse_metrics(result_line)
                            results[domain] = metrics
                
                return results
    except Exception as e:
        print(f"Error reading fold {fold_num} results: {e}")
    
    return None

def main():
    """Run all 5 folds and calculate average metrics"""
    all_results = {}
    
    for fold_num in range(1, 6):
        success = run_fold(fold_num)
        if not success:
            print(f"Warning: Fold {fold_num} may have encountered issues")
        else:
            # Extract results
            results = extract_fold_results(fold_num)
            if results:
                all_results[f'Fold_{fold_num}'] = results
    
    # Calculate and save average metrics
    print(f"\n{'='*60}")
    print("Calculating Average Metrics")
    print(f"{'='*60}\n")
    
    summary_file = '../Data/results_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("5-Fold Cross Validation Results\n")
        f.write("="*60 + "\n\n")
        
        # Write individual fold results
        for fold_name, results in all_results.items():
            f.write(f"\n{fold_name}:\n")
            f.write("-" * 40 + "\n")
            if results:
                for domain, metrics in results.items():
                    if metrics:
                        f.write(f"  {domain}:\n")
                        f.write(f"    Hit@10: {metrics['hits'][0] if metrics['hits'] else 'N/A':.4f}\n")
                        f.write(f"    NDCG@10: {metrics['ndcgs'][0] if metrics['ndcgs'] else 'N/A':.4f}\n")
            else:
                f.write("  No metrics found\n")
        
        # Calculate and write average metrics
        f.write("\n" + "="*60 + "\n")
        f.write("AVERAGE METRICS (5 Folds)\n")
        f.write("="*60 + "\n\n")
        
        # Aggregate source and target metrics
        source_hits = []
        source_ndcgs = []
        target_hits = []
        target_ndcgs = []
        
        for fold_name, results in all_results.items():
            if results:
                if 'source result' in results and results['source result']:
                    source_hits.extend(results['source result']['hits'])
                    source_ndcgs.extend(results['source result']['ndcgs'])
                if 'target result' in results and results['target result']:
                    target_hits.extend(results['target result']['hits'])
                    target_ndcgs.extend(results['target result']['ndcgs'])
        
        if source_hits:
            f.write(f"Source Domain:\n")
            f.write(f"  Avg Hit@10: {np.mean([source_hits[i] for i in range(0, len(source_hits), len(source_hits)//5 if len(source_hits)//5 > 0 else 1)]):.4f}\n")
            f.write(f"  Avg NDCG@10: {np.mean([source_ndcgs[i] for i in range(0, len(source_ndcgs), len(source_ndcgs)//5 if len(source_ndcgs)//5 > 0 else 1)]):.4f}\n\n")
        
        if target_hits:
            f.write(f"Target Domain:\n")
            f.write(f"  Avg Hit@10: {np.mean([target_hits[i] for i in range(0, len(target_hits), len(target_hits)//5 if len(target_hits)//5 > 0 else 1)]):.4f}\n")
            f.write(f"  Avg NDCG@10: {np.mean([target_ndcgs[i] for i in range(0, len(target_ndcgs), len(target_ndcgs)//5 if len(target_ndcgs)//5 > 0 else 1)]):.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Individual fold results saved in: ./output/folds/\n")
        f.write("="*60 + "\n")
    
    # print(f"✓ Summary saved to {summary_file}")
    print("\nResults structure:")
    print("  ./output/")
    print("    ├── miRNA-disease_miRNA-target.result (all folds combined)")
    print("    └── folds/")
    print("        ├── Fold_1_result.txt")
    print("        ├── Fold_2_result.txt")
    print("        ├── Fold_3_result.txt")
    print("        ├── Fold_4_result.txt")
    print("        └── Fold_5_result.txt")
    # print(f"\n✓ Full summary: {summary_file}")

if __name__ == '__main__':
    main()


