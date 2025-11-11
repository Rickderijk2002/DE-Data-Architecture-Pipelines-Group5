"""
Data Quality Pipeline for Netflix-Style Dataset
Steps:
1. Data Loading and Basic Information
2. Missing Values Analysis
3. Duplicate Detection
"""

import pandas as pd
import os
from pathlib import Path


def load_datasets(data_dir="Data/raw"):
    """
    Load all datasets from the Data/raw directory.
    
    Returns:
        dict: Dictionary with dataset names as keys and DataFrames as values
    """
    datasets = {}
    data_path = Path(data_dir)
    
    # List of all dataset files
    dataset_files = [
        "users.csv",
        "movies.csv", 
        "watch_history.csv",
        "recommendation_logs.csv",
        "search_logs.csv",
        "reviews.csv"
    ]
    
    print("=" * 60)
    print("STEP 1: Loading Datasets")
    print("=" * 60)
    
    for file in dataset_files:
        file_path = data_path / file
        if file_path.exists():
            print(f"Loading {file}...")
            df = pd.read_csv(file_path)
            dataset_name = file.replace(".csv", "")
            datasets[dataset_name] = df
            print(f"  [OK] Loaded: {len(df)} rows, {len(df.columns)} columns")
        else:
            print(f"  [WARNING] {file} not found at {file_path}")
    
    return datasets


def get_basic_info(datasets):
    """
    Print basic information about all loaded datasets.
    
    Args:
        datasets: Dictionary of dataset names and DataFrames
    """
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    
    for name, df in datasets.items():
        print(f"\n{name.upper()}:")
        print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"  Columns: {', '.join(df.columns.tolist())}")


def analyze_missing_values(datasets):
    """
    Analyze missing values across all datasets.
    
    Args:
        datasets: Dictionary of dataset names and DataFrames
    
    Returns:
        dict: Dictionary with missing value statistics per dataset
    """
    print("\n" + "=" * 60)
    print("STEP 2: Missing Values Analysis")
    print("=" * 60)
    
    missing_stats = {}
    
    for name, df in datasets.items():
        print(f"\n{name.upper()}:")
        
        # Calculate missing values
        missing_count = df.isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        # Filter to only columns with missing values
        missing_cols = missing_count[missing_count > 0]
        
        if len(missing_cols) > 0:
            print(f"  Total rows: {len(df):,}")
            print(f"  Columns with missing values: {len(missing_cols)}")
            print(f"  Total missing values: {missing_count.sum():,}")
            print(f"  Overall missing percentage: {(missing_count.sum() / (len(df) * len(df.columns))) * 100:.2f}%")
            print(f"\n  Missing values by column:")
            
            # Sort by missing count (descending)
            for col in missing_cols.sort_values(ascending=False).index:
                count = missing_count[col]
                pct = missing_pct[col]
                print(f"    - {col}: {count:,} ({pct:.2f}%)")
        else:
            print(f"  [OK] No missing values found")
        
        # Store statistics
        missing_stats[name] = {
            'total_rows': len(df),
            'total_missing': missing_count.sum(),
            'columns_with_missing': len(missing_cols),
            'missing_by_column': missing_count[missing_count > 0].to_dict()
        }
    
    return missing_stats


def detect_duplicates(datasets):
    """
    Detect duplicate rows across all datasets.
    Checks both complete duplicates and duplicates based on key identifier columns.
    
    Args:
        datasets: Dictionary of dataset names and DataFrames
    
    Returns:
        dict: Dictionary with duplicate statistics per dataset
    """
    print("\n" + "=" * 60)
    print("STEP 3: Duplicate Detection")
    print("=" * 60)
    
    # Define key identifier columns for each dataset
    key_columns = {
        'users': ['user_id', 'email'],
        'movies': ['movie_id'],
        'watch_history': ['session_id'],
        'recommendation_logs': ['recommendation_id'],
        'search_logs': ['search_id'],
        'reviews': ['review_id']
    }
    
    duplicate_stats = {}
    
    for name, df in datasets.items():
        print(f"\n{name.upper()}:")
        print(f"  Total rows: {len(df):,}")
        
        # Check for complete duplicates (all columns match)
        complete_duplicates = df.duplicated(keep=False)
        complete_dup_count = complete_duplicates.sum()
        complete_dup_pct = (complete_dup_count / len(df)) * 100
        
        # Check for duplicates based on key columns
        key_dup_count = 0
        key_dup_pct = 0.0
        key_cols_used = []
        
        if name in key_columns:
            # Check which key columns exist in the dataframe
            available_key_cols = [col for col in key_columns[name] if col in df.columns]
            
            if available_key_cols:
                key_cols_used = available_key_cols
                key_duplicates = df.duplicated(subset=available_key_cols, keep=False)
                key_dup_count = key_duplicates.sum()
                key_dup_pct = (key_dup_count / len(df)) * 100
        
        # Report complete duplicates
        if complete_dup_count > 0:
            unique_duplicate_rows = df[complete_duplicates].drop_duplicates().shape[0]
            print(f"  Complete duplicates (all columns):")
            print(f"    - Duplicate rows: {complete_dup_count:,} ({complete_dup_pct:.2f}%)")
            print(f"    - Unique duplicate groups: {unique_duplicate_rows:,}")
        else:
            print(f"  [OK] No complete duplicates found")
        
        # Report key column duplicates
        if key_dup_count > 0:
            print(f"  Key column duplicates ({', '.join(key_cols_used)}):")
            print(f"    - Duplicate rows: {key_dup_count:,} ({key_dup_pct:.2f}%)")
            # Show some examples
            if key_cols_used:
                dup_examples = df[df.duplicated(subset=key_cols_used, keep=False)].groupby(key_cols_used).size().sort_values(ascending=False).head(3)
                if len(dup_examples) > 0:
                    print(f"    - Top duplicate groups:")
                    for idx, count in dup_examples.items():
                        if len(key_cols_used) == 1:
                            print(f"      * {key_cols_used[0]}={idx}: {count} occurrences")
                        else:
                            print(f"      * {dict(zip(key_cols_used, idx))}: {count} occurrences")
        else:
            if key_cols_used:
                print(f"  [OK] No duplicates found in key columns ({', '.join(key_cols_used)})")
        
        # Store statistics
        duplicate_stats[name] = {
            'total_rows': len(df),
            'complete_duplicates': complete_dup_count,
            'complete_duplicates_pct': complete_dup_pct,
            'key_column_duplicates': key_dup_count,
            'key_column_duplicates_pct': key_dup_pct,
            'key_columns_checked': key_cols_used
        }
    
    return duplicate_stats


def save_processed_data(datasets, output_dir="Data/processed"):
    """
    Save all processed datasets to the output directory.
    
    Args:
        datasets: Dictionary of dataset names and DataFrames
        output_dir: Directory path to save processed data
    """
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Saving Processed Data")
    print("=" * 60)
    
    for name, df in datasets.items():
        output_file = output_path / f"{name}_processed.csv"
        df.to_csv(output_file, index=False)
        print(f"  [OK] Saved {name}_processed.csv: {len(df):,} rows")
    
    print(f"\n[OK] All datasets saved to: {output_path.absolute()}")


if __name__ == "__main__":
    # Load all datasets
    data = load_datasets()
    
    # Display basic information
    get_basic_info(data)
    
    # Analyze missing values
    missing_stats = analyze_missing_values(data)
    
    # Detect duplicates
    duplicate_stats = detect_duplicates(data)
    
    # Save processed data
    save_processed_data(data)
    
    print("\n" + "=" * 60)
    print("Pipeline Complete: Data loaded, analyzed, and saved successfully!")
    print("=" * 60)

