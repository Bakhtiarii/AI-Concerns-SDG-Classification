"""
AI Concerns to SDG Classification Pipeline
===========================================
This script implements a complete pipeline for classifying AI-related concerns from Twitter/X
data to UN Sustainable Development Goals (SDGs) using transformer-based models.

Author: Mohammadreza
Institution: Stony Brook University
Paper: Mapping AI Concerns to Sustainable Development Goals using Multi-label Classification

Usage:
    1. Update the file paths in the CONFIGURATION section
    2. Run each section sequentially or use specific functions as needed
"""

import os
import re
import json
import glob
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModel
)
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR ENVIRONMENT
# ============================================================================

# Input/Output Directory Paths
DATA_DIR = './data'                          # Main data directory
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw') # Raw data files
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')  # Processed data
RESULTS_DIR = './results'                    # Classification results
PLOTS_DIR = './plots'                        # Visualization outputs

# Specific File Paths (examples - update as needed)
INPUT_CSV = os.path.join(RAW_DATA_DIR, 'tweets_raw.csv')
INPUT_JSON = os.path.join(RAW_DATA_DIR, 'tweets_raw.json')
COMBINED_CSV = os.path.join(PROCESSED_DATA_DIR, 'combined_data.csv')
PREPROCESSED_CSV = os.path.join(PROCESSED_DATA_DIR, 'preprocessed_data.csv')
CLASSIFIED_CSV = os.path.join(RESULTS_DIR, 'sdg_classified_results.csv')
FILTERED_CSV = os.path.join(RESULTS_DIR, 'sdg_classified_filtered.csv')
PLOT_OUTPUT = os.path.join(PLOTS_DIR, 'sdg_distribution.png')

# Model Configuration
BERT_MODEL_NAME = "sadickam/sdg-classification-bert"
SENTENCE_TRANSFORMER_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# Processing Parameters
MIN_WORDS = 5              # Minimum words per tweet
MIN_CHARS = 50             # Minimum characters per tweet
CLASSIFICATION_THRESHOLD = 0.3  # Confidence threshold for SDG classification
BATCH_SIZE = 16            # Batch size for model inference

# SDG Labels
SDG_LABELS = [
    'No Poverty', 'Zero Hunger', 'Good Health and Well-being',
    'Quality Education', 'Gender Equality', 'Clean Water and Sanitation',
    'Affordable and Clean Energy', 'Decent Work and Economic Growth',
    'Industry, Innovation and Infrastructure', 'Reduced Inequalities',
    'Sustainable Cities and Communities', 'Responsible Consumption and Production',
    'Climate Action', 'Life Below Water', 'Life on Land',
    'Peace, Justice and Strong Institutions', 'Partnerships for the Goals'
]

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================================
# SECTION 1: DATA LOADING AND COMBINATION
# ============================================================================

def combine_csv_files(input_pattern, output_path):
    """
    Combine multiple CSV files into a single DataFrame.
    
    Args:
        input_pattern (str): Glob pattern for input CSV files (e.g., './data/*.csv')
        output_path (str): Path to save the combined CSV file
    
    Returns:
        pd.DataFrame: Combined DataFrame
    """
    print("=" * 70)
    print("COMBINING CSV FILES")
    print("=" * 70)
    
    csv_files = glob.glob(input_pattern)
    print(f"Found {len(csv_files)} CSV files to combine")
    
    if not csv_files:
        print(f"Warning: No files found matching pattern: {input_pattern}")
        return None
    
    dataframes = []
    for file in tqdm(csv_files, desc="Reading CSV files"):
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df.to_csv(output_path, index=False)
        print(f"\nCombined {len(dataframes)} files into {output_path}")
        print(f"Total rows: {len(combined_df)}")
        return combined_df
    else:
        print("No data to combine")
        return None


def json_to_csv(json_path, csv_path, required_fields=None):
    """
    Convert JSON file to CSV, extracting specified fields.
    
    Args:
        json_path (str): Path to input JSON file
        csv_path (str): Path to output CSV file
        required_fields (list): List of field names to extract
    
    Returns:
        pd.DataFrame: Converted DataFrame
    """
    print("=" * 70)
    print("CONVERTING JSON TO CSV")
    print("=" * 70)
    
    if required_fields is None:
        required_fields = ['text', 'tweet.created_at', 'tweet.user_id_str', 'tweet.id_str']
    
    print(f"Loading JSON file: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        print(f"Processing {len(data)} records...")
        
        # Extract data from each record
        extracted_data = {field: [] for field in required_fields}
        
        for record in tqdm(data, desc="Extracting fields"):
            tweet_data = record.get('tweet', {})
            
            for field in required_fields:
                if '.' in field:
                    # Handle nested fields
                    parts = field.split('.')
                    value = record
                    for part in parts:
                        value = value.get(part, {}) if isinstance(value, dict) else ''
                    extracted_data[field].append(value if value else '')
                else:
                    extracted_data[field].append(record.get(field, ''))
        
        # Create DataFrame
        df = pd.DataFrame(extracted_data)
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"\nConversion complete!")
        print(f"CSV file saved to: {csv_path}")
        print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        print(f"Column names: {df.columns.tolist()}")
        
        return df
        
    except Exception as e:
        print(f"Error converting JSON to CSV: {e}")
        return None


# ============================================================================
# SECTION 2: DATA PREPROCESSING
# ============================================================================

def initial_preprocessing(df, text_column='text'):
    """
    Perform initial preprocessing: convert to string, handle NaNs, remove duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Name of the text column to process
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    print("\n--- Initial Preprocessing ---")
    initial_rows = len(df)
    print(f"Initial number of rows: {initial_rows}")
    
    # Convert all entries to strings
    df[text_column] = df[text_column].astype(str)
    
    # Replace NaNs with empty strings
    df[text_column] = df[text_column].fillna('')
    
    # Remove duplicates
    df = df.drop_duplicates(subset=[text_column])
    print(f"Rows after removing duplicates: {len(df)} (removed {initial_rows - len(df)})")
    
    return df


def clean_text(text):
    """
    Clean text by removing URLs, mentions, hashtags, and converting to lowercase.
    
    Args:
        text (str): Input text
    
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def filter_short_texts(df, text_column='cleaned_text', min_words=MIN_WORDS, min_chars=MIN_CHARS):
    """
    Filter out texts that are too short.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Name of the text column to filter
        min_words (int): Minimum number of words
        min_chars (int): Minimum number of characters
    
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    initial_rows = len(df)
    df = df[
        (df[text_column].apply(lambda x: len(x.split()) >= min_words)) &
        (df[text_column].apply(lambda x: len(x) >= min_chars))
    ]
    print(f"Rows after filtering short texts: {len(df)} (removed {initial_rows - len(df)})")
    return df


def preprocess_data(input_path, output_path, text_column='text'):
    """
    Complete preprocessing pipeline for tweet data.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to save preprocessed CSV file
        text_column (str): Name of the text column
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    print("=" * 70)
    print("DATA PREPROCESSING")
    print("=" * 70)
    
    # Read the CSV file
    print(f"Reading file: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Initial dataset size: {len(df)} rows")
    
    # Apply initial preprocessing
    df = initial_preprocessing(df, text_column)
    
    # Create cleaned text column
    print("\n--- Text Cleaning ---")
    df['cleaned_text'] = df[text_column].apply(clean_text)
    
    # Filter short texts
    df = filter_short_texts(df)
    
    # Print statistics
    print("\n--- Text Length Statistics ---")
    df['original_length'] = df[text_column].str.len()
    df['cleaned_length'] = df['cleaned_text'].str.len()
    
    print("\nOriginal text length:")
    print(df['original_length'].describe())
    print("\nCleaned text length:")
    print(df['cleaned_length'].describe())
    
    # Display sample
    print("\n--- Sample of Processed Data ---")
    print(df[[text_column, 'cleaned_text']].head(3))
    
    # Remove temporary columns and save
    df = df.drop(['original_length', 'cleaned_length'], axis=1)
    df.to_csv(output_path, index=False)
    print(f"\nPreprocessed data saved to: {output_path}")
    print(f"Final dataset size: {len(df)} rows")
    
    return df


# ============================================================================
# SECTION 3: SDG CLASSIFICATION WITH BERT
# ============================================================================

class SDGClassifier:
    """
    SDG Classification using fine-tuned BERT model.
    """
    
    def __init__(self, model_name=BERT_MODEL_NAME, threshold=CLASSIFICATION_THRESHOLD):
        """
        Initialize the SDG classifier.
        
        Args:
            model_name (str): HuggingFace model name
            threshold (float): Classification confidence threshold
        """
        print("=" * 70)
        print("INITIALIZING SDG CLASSIFIER")
        print("=" * 70)
        print(f"Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Classification threshold: {self.threshold}")
    
    def classify_text(self, text):
        """
        Classify a single text to SDG labels.
        
        Args:
            text (str): Input text
        
        Returns:
            dict: Dictionary with SDG predictions and probabilities
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.sigmoid(outputs.logits)[0].cpu().numpy()
        
        # Get predictions above threshold
        predictions = {}
        for idx, prob in enumerate(probabilities):
            if prob >= self.threshold:
                sdg_num = idx + 1
                predictions[f'SDG_{sdg_num}'] = float(prob)
        
        return predictions
    
    def classify_batch(self, texts, batch_size=BATCH_SIZE):
        """
        Classify multiple texts in batches.
        
        Args:
            texts (list): List of texts to classify
            batch_size (int): Batch size for processing
        
        Returns:
            list: List of prediction dictionaries
        """
        all_predictions = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Classifying texts"):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.sigmoid(outputs.logits).cpu().numpy()
            
            # Process each text in the batch
            for probs in probabilities:
                predictions = {}
                for idx, prob in enumerate(probs):
                    if prob >= self.threshold:
                        sdg_num = idx + 1
                        predictions[f'SDG_{sdg_num}'] = float(prob)
                all_predictions.append(predictions)
        
        return all_predictions
    
    def classify_dataframe(self, df, text_column='cleaned_text', output_path=None):
        """
        Classify all texts in a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Name of the text column to classify
            output_path (str): Optional path to save results
        
        Returns:
            pd.DataFrame: DataFrame with SDG classifications
        """
        print("\n--- Classifying DataFrame ---")
        print(f"Processing {len(df)} texts...")
        
        texts = df[text_column].tolist()
        predictions = self.classify_batch(texts)
        
        # Add SDG columns to DataFrame
        for i in range(1, 18):
            sdg_col = f'SDG_{i}'
            df[sdg_col] = [pred.get(sdg_col, 0.0) for pred in predictions]
        
        # Add column for number of assigned SDGs
        sdg_columns = [f'SDG_{i}' for i in range(1, 18)]
        df['num_sdgs'] = (df[sdg_columns] > 0).sum(axis=1)
        
        # Filter rows with at least one SDG assignment
        df_classified = df[df['num_sdgs'] > 0].copy()
        
        print(f"\nClassification complete!")
        print(f"Texts with SDG assignments: {len(df_classified)} ({len(df_classified)/len(df)*100:.1f}%)")
        print(f"Texts without SDG assignments: {len(df) - len(df_classified)}")
        
        if output_path:
            df_classified.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")
        
        return df_classified


# ============================================================================
# SECTION 4: ZERO-SHOT CLASSIFICATION (Alternative Method)
# ============================================================================

class ZeroShotSDGClassifier:
    """
    Zero-shot SDG classification using sentence transformers.
    """
    
    def __init__(self, model_name=SENTENCE_TRANSFORMER_MODEL, threshold=0.3):
        """
        Initialize zero-shot classifier.
        
        Args:
            model_name (str): Sentence transformer model name
            threshold (float): Similarity threshold
        """
        print("=" * 70)
        print("INITIALIZING ZERO-SHOT CLASSIFIER")
        print("=" * 70)
        print(f"Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # SDG descriptions for zero-shot classification
        self.sdg_descriptions = [
            "ending poverty in all its forms everywhere",
            "ending hunger achieving food security and promoting sustainable agriculture",
            "ensuring healthy lives and promoting well-being for all ages",
            "ensuring inclusive and equitable quality education and lifelong learning",
            "achieving gender equality and empowering all women and girls",
            "ensuring availability and sustainable management of water and sanitation",
            "ensuring access to affordable reliable sustainable and modern energy",
            "promoting sustained inclusive and sustainable economic growth employment",
            "building resilient infrastructure promoting sustainable industrialization",
            "reducing inequality within and among countries",
            "making cities and human settlements inclusive safe resilient and sustainable",
            "ensuring sustainable consumption and production patterns",
            "taking urgent action to combat climate change and its impacts",
            "conserving and sustainably using oceans seas and marine resources",
            "protecting restoring and promoting sustainable use of terrestrial ecosystems",
            "promoting peaceful and inclusive societies access to justice",
            "strengthening means of implementation and revitalizing global partnerships"
        ]
        
        print(f"Model loaded successfully on {self.device}")
    
    def get_embeddings(self, texts):
        """Get embeddings for texts."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return embeddings
    
    def classify_text(self, text):
        """Classify text using zero-shot approach."""
        # Get embeddings
        text_embedding = self.get_embeddings([text])
        sdg_embeddings = self.get_embeddings(self.sdg_descriptions)
        
        # Calculate similarities
        similarities = cosine_similarity(text_embedding, sdg_embeddings)[0]
        
        # Get predictions above threshold
        predictions = {}
        for idx, similarity in enumerate(similarities):
            if similarity >= self.threshold:
                sdg_num = idx + 1
                predictions[f'SDG_{sdg_num}'] = float(similarity)
        
        return predictions


# ============================================================================
# SECTION 5: RESULTS ANALYSIS AND VISUALIZATION
# ============================================================================

def analyze_sdg_distribution(df, output_path=None):
    """
    Analyze and visualize SDG distribution in classified data.
    
    Args:
        df (pd.DataFrame): DataFrame with SDG classifications
        output_path (str): Optional path to save plot
    
    Returns:
        pd.Series: SDG distribution counts
    """
    print("=" * 70)
    print("SDG DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    sdg_columns = [f'SDG_{i}' for i in range(1, 18)]
    
    # Count texts assigned to each SDG
    sdg_counts = {}
    for i, sdg_col in enumerate(sdg_columns, 1):
        count = (df[sdg_col] > 0).sum()
        sdg_counts[f'SDG {i}'] = count
    
    sdg_series = pd.Series(sdg_counts)
    
    # Print statistics
    print("\nSDG Distribution:")
    print(sdg_series)
    print(f"\nTotal texts: {len(df)}")
    print(f"Average SDGs per text: {df['num_sdgs'].mean():.2f}")
    print(f"Max SDGs per text: {df['num_sdgs'].max()}")
    
    # Create visualization
    plt.figure(figsize=(14, 8))
    
    # Calculate percentages
    percentages = (sdg_series / len(df) * 100)
    
    # Create bar plot
    bars = plt.bar(range(len(sdg_series)), percentages, color='steelblue', edgecolor='black')
    
    # Add percentage labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Sustainable Development Goals (SDGs)', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage of Texts (%)', fontsize=12, fontweight='bold')
    plt.title('Distribution of AI Concerns Across 17 SDGs', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(range(len(sdg_series)), [f'SDG\n{i+1}' for i in range(17)], fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    
    plt.show()
    
    return sdg_series


def generate_sdg_summary_report(df, output_path=None):
    """
    Generate a comprehensive summary report of SDG classifications.
    
    Args:
        df (pd.DataFrame): DataFrame with SDG classifications
        output_path (str): Optional path to save report
    
    Returns:
        dict: Summary statistics
    """
    print("=" * 70)
    print("GENERATING SDG SUMMARY REPORT")
    print("=" * 70)
    
    sdg_columns = [f'SDG_{i}' for i in range(1, 18)]
    
    summary = {
        'total_texts': len(df),
        'sdg_distribution': {},
        'top_5_sdgs': [],
        'bottom_5_sdgs': [],
        'multi_label_stats': {
            'avg_sdgs_per_text': df['num_sdgs'].mean(),
            'max_sdgs_per_text': df['num_sdgs'].max(),
            'min_sdgs_per_text': df['num_sdgs'].min(),
            'texts_with_single_sdg': (df['num_sdgs'] == 1).sum(),
            'texts_with_multiple_sdgs': (df['num_sdgs'] > 1).sum()
        }
    }
    
    # Calculate SDG distribution
    for i, sdg_col in enumerate(sdg_columns, 1):
        count = (df[sdg_col] > 0).sum()
        percentage = (count / len(df)) * 100
        summary['sdg_distribution'][f'SDG_{i}'] = {
            'count': int(count),
            'percentage': round(percentage, 2),
            'label': SDG_LABELS[i-1]
        }
    
    # Get top and bottom SDGs
    sdg_counts = [(i, (df[f'SDG_{i}'] > 0).sum()) for i in range(1, 18)]
    sdg_counts_sorted = sorted(sdg_counts, key=lambda x: x[1], reverse=True)
    
    summary['top_5_sdgs'] = [
        {'sdg': i, 'count': count, 'label': SDG_LABELS[i-1]} 
        for i, count in sdg_counts_sorted[:5]
    ]
    
    summary['bottom_5_sdgs'] = [
        {'sdg': i, 'count': count, 'label': SDG_LABELS[i-1]} 
        for i, count in sdg_counts_sorted[-5:]
    ]
    
    # Print summary
    print(f"\nTotal texts analyzed: {summary['total_texts']}")
    print(f"\nMulti-label Statistics:")
    print(f"  - Average SDGs per text: {summary['multi_label_stats']['avg_sdgs_per_text']:.2f}")
    print(f"  - Texts with single SDG: {summary['multi_label_stats']['texts_with_single_sdg']}")
    print(f"  - Texts with multiple SDGs: {summary['multi_label_stats']['texts_with_multiple_sdgs']}")
    
    print(f"\nTop 5 SDGs:")
    for item in summary['top_5_sdgs']:
        print(f"  SDG {item['sdg']}: {item['label']} - {item['count']} texts")
    
    print(f"\nBottom 5 SDGs:")
    for item in summary['bottom_5_sdgs']:
        print(f"  SDG {item['sdg']}: {item['label']} - {item['count']} texts")
    
    # Save report to JSON if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nReport saved to: {output_path}")
    
    return summary


# ============================================================================
# SECTION 6: FILTERING AND EXTRACTION
# ============================================================================

def filter_by_query_pattern(input_path, output_path, query_patterns=None):
    """
    Filter data based on query patterns (e.g., specific AI concern keywords).
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to save filtered CSV
        query_patterns (list): List of regex patterns to match
    
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    print("=" * 70)
    print("FILTERING BY QUERY PATTERNS")
    print("=" * 70)
    
    if query_patterns is None:
        query_patterns = [
            r'"Artificial Intelligence"\s+AND\s+"concern"',
            r'"\s+AI\s+"\s+AND\s+concern',
            r'"Artificial Intelligence"\s+AND\s+worries',
            r'"\s+AI\s+"\s+AND\s+worries'
        ]
    
    print(f"Query patterns: {query_patterns}")
    
    # Read data
    df = pd.read_csv(input_path)
    print(f"Initial rows: {len(df)}")
    
    # Check if query column exists
    if 'query' not in df.columns:
        print("Warning: 'query' column not found. Skipping filtering.")
        return df
    
    # Filter by patterns
    def matches_pattern(query_text):
        if pd.isna(query_text):
            return False
        for pattern in query_patterns:
            if re.search(pattern, str(query_text), re.IGNORECASE):
                return True
        return False
    
    filtered_df = df[df['query'].apply(matches_pattern)]
    
    # Save results
    filtered_df.to_csv(output_path, index=False)
    print(f"Filtered rows: {len(filtered_df)}")
    print(f"Results saved to: {output_path}")
    
    return filtered_df


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def run_full_pipeline():
    """
    Execute the complete SDG classification pipeline.
    """
    print("\n" + "=" * 70)
    print("STARTING FULL SDG CLASSIFICATION PIPELINE")
    print("=" * 70 + "\n")
    
    # Step 1: Combine CSV files (if needed)
    # Uncomment if you have multiple CSV files to combine
    # combined_df = combine_csv_files(
    #     input_pattern=os.path.join(RAW_DATA_DIR, '*.csv'),
    #     output_path=COMBINED_CSV
    # )
    
    # Step 2: Convert JSON to CSV (if needed)
    # Uncomment if you have JSON data
    # df = json_to_csv(
    #     json_path=INPUT_JSON,
    #     csv_path=INPUT_CSV
    # )
    
    # Step 3: Preprocess data
    print("\n" + "=" * 70)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 70)
    df_preprocessed = preprocess_data(
        input_path=INPUT_CSV,
        output_path=PREPROCESSED_CSV
    )
    
    # Step 4: Classify with BERT
    print("\n" + "=" * 70)
    print("STEP 2: SDG CLASSIFICATION")
    print("=" * 70)
    classifier = SDGClassifier()
    df_classified = classifier.classify_dataframe(
        df=df_preprocessed,
        text_column='cleaned_text',
        output_path=CLASSIFIED_CSV
    )
    
    # Step 5: Filter results (optional)
    # Uncomment if you want to apply additional filtering
    # df_filtered = filter_by_query_pattern(
    #     input_path=CLASSIFIED_CSV,
    #     output_path=FILTERED_CSV
    # )
    
    # Step 6: Analyze and visualize results
    print("\n" + "=" * 70)
    print("STEP 3: RESULTS ANALYSIS")
    print("=" * 70)
    analyze_sdg_distribution(
        df=df_classified,
        output_path=PLOT_OUTPUT
    )
    
    # Step 7: Generate summary report
    summary = generate_sdg_summary_report(
        df=df_classified,
        output_path=os.path.join(RESULTS_DIR, 'summary_report.json')
    )
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Visualizations saved to: {PLOTS_DIR}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Main entry point for the script.
    
    Usage examples:
    
    1. Run full pipeline:
        python sdg_classification_pipeline.py
    
    2. Run specific functions:
        # Preprocess data only
        df = preprocess_data(INPUT_CSV, PREPROCESSED_CSV)
        
        # Classify only
        classifier = SDGClassifier()
        df_classified = classifier.classify_dataframe(df, output_path=CLASSIFIED_CSV)
        
        # Analyze only
        analyze_sdg_distribution(df_classified, PLOT_OUTPUT)
    """
    
    # Run the full pipeline
    run_full_pipeline()
    
    # Alternative: Run individual components
    # Uncomment the functions you want to run individually:
    
    # # Preprocessing only
    # df = preprocess_data(INPUT_CSV, PREPROCESSED_CSV)
    
    # # Classification only
    # classifier = SDGClassifier()
    # df_classified = classifier.classify_dataframe(
    #     pd.read_csv(PREPROCESSED_CSV),
    #     output_path=CLASSIFIED_CSV
    # )
    
    # # Analysis only
    # df_classified = pd.read_csv(CLASSIFIED_CSV)
    # analyze_sdg_distribution(df_classified, PLOT_OUTPUT)
    # generate_sdg_summary_report(df_classified, 
    #     os.path.join(RESULTS_DIR, 'summary_report.json'))
