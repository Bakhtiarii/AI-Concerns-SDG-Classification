# Configuration Template
# Copy this file to config.py and update with your specific paths

# ==============================================================================
# DIRECTORY PATHS
# ==============================================================================

# Main directories
DATA_DIR = './data'                          # Main data directory
RAW_DATA_DIR = './data/raw'                  # Raw data files (CSV, JSON)
PROCESSED_DATA_DIR = './data/processed'      # Preprocessed data
RESULTS_DIR = './results'                    # Classification results
PLOTS_DIR = './plots'                        # Visualization outputs

# ==============================================================================
# FILE PATHS
# ==============================================================================

# Input files
INPUT_CSV = './data/raw/tweets_raw.csv'             # Main input CSV file
INPUT_JSON = './data/raw/tweets_raw.json'           # Alternative JSON input
MULTIPLE_CSV_PATTERN = './data/raw/*.csv'           # Pattern for multiple CSVs

# Output files
COMBINED_CSV = './data/processed/combined_data.csv'
PREPROCESSED_CSV = './data/processed/preprocessed_data.csv'
CLASSIFIED_CSV = './results/sdg_classified_results.csv'
FILTERED_CSV = './results/sdg_classified_filtered.csv'
SUMMARY_REPORT = './results/summary_report.json'

# Visualization outputs
PLOT_OUTPUT = './plots/sdg_distribution.png'
CONFUSION_MATRIX_PLOT = './plots/confusion_matrix.png'

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

# HuggingFace Model Names
BERT_MODEL_NAME = "sadickam/sdg-classification-bert"
SENTENCE_TRANSFORMER_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
XLNET_MODEL_NAME = "xlnet-base-cased"

# Model inference settings
BATCH_SIZE = 16              # Batch size for model inference
MAX_LENGTH = 512             # Maximum sequence length for tokenization
USE_GPU = True               # Use GPU if available

# ==============================================================================
# PREPROCESSING PARAMETERS
# ==============================================================================

# Text filtering thresholds
MIN_WORDS = 5                # Minimum words per tweet
MIN_CHARS = 50               # Minimum characters per tweet

# Duplicate handling
REMOVE_DUPLICATES = True     # Remove duplicate tweets
DUPLICATE_COLUMN = 'text'    # Column to check for duplicates

# Text cleaning options
CONVERT_LOWERCASE = True     # Convert text to lowercase
REMOVE_URLS = True           # Remove URLs from text
REMOVE_MENTIONS = True       # Remove @mentions from text
REMOVE_HASHTAGS = True       # Remove hashtag symbols (#)
REMOVE_EXTRA_SPACES = True   # Remove extra whitespace

# ==============================================================================
# CLASSIFICATION PARAMETERS
# ==============================================================================

# Confidence thresholds
CLASSIFICATION_THRESHOLD = 0.3   # Minimum confidence for SDG assignment
HIGH_CONFIDENCE_THRESHOLD = 0.7  # Threshold for high-confidence predictions

# Multi-label settings
ALLOW_MULTIPLE_SDGS = True       # Allow texts to have multiple SDG labels
MAX_SDGS_PER_TEXT = 5            # Maximum SDGs to assign per text (None = unlimited)

# ==============================================================================
# SDG LABELS AND DESCRIPTIONS
# ==============================================================================

SDG_LABELS = [
    'No Poverty',
    'Zero Hunger',
    'Good Health and Well-being',
    'Quality Education',
    'Gender Equality',
    'Clean Water and Sanitation',
    'Affordable and Clean Energy',
    'Decent Work and Economic Growth',
    'Industry, Innovation and Infrastructure',
    'Reduced Inequalities',
    'Sustainable Cities and Communities',
    'Responsible Consumption and Production',
    'Climate Action',
    'Life Below Water',
    'Life on Land',
    'Peace, Justice and Strong Institutions',
    'Partnerships for the Goals'
]

# SDG descriptions for zero-shot classification
SDG_DESCRIPTIONS = [
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

# ==============================================================================
# DATA COLLECTION PARAMETERS (if using Twitter API)
# ==============================================================================

# Query keywords for data collection
QUERY_KEYWORDS = [
    '"artificial intelligence" AND concern',
    '"AI" AND concern',
    '"artificial intelligence" AND risks',
    '"AI" AND risks',
    '"artificial intelligence" AND worries',
    '"AI" AND worries',
    '"artificial intelligence" AND fear',
    '"AI" AND fear'
]

# Date range for data collection
START_DATE = "2022-01-01"
END_DATE = "2025-01-31"

# ==============================================================================
# VISUALIZATION SETTINGS
# ==============================================================================

# Plot settings
FIGURE_SIZE = (14, 8)        # Default figure size (width, height)
DPI = 300                    # Resolution for saved plots
PLOT_STYLE = 'seaborn-v0_8'  # Matplotlib style
COLOR_PALETTE = 'steelblue'  # Color for plots

# Font settings
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10

# ==============================================================================
# LOGGING AND OUTPUT
# ==============================================================================

# Logging
VERBOSE = True               # Print detailed progress information
LOG_FILE = './logs/pipeline.log'  # Log file path (None to disable)

# Output format
SAVE_INTERMEDIATE_FILES = True   # Save intermediate processing results
OUTPUT_FORMAT = 'csv'            # Format for output files: 'csv', 'json', 'both'

# ==============================================================================
# PERFORMANCE OPTIMIZATION
# ==============================================================================

# Parallel processing
USE_MULTIPROCESSING = False  # Use multiprocessing for data processing
NUM_WORKERS = 4              # Number of worker processes

# Memory management
CHUNK_SIZE = 10000           # Process data in chunks of this size
LOW_MEMORY_MODE = False      # Use low-memory processing mode

# ==============================================================================
# EVALUATION PARAMETERS (if using labeled data)
# ==============================================================================

# Test set configuration
TEST_SIZE = 0.2              # Proportion of data for testing
RANDOM_SEED = 42             # Random seed for reproducibility

# Evaluation metrics
EVAL_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score'
]

# ==============================================================================
# NOTES
# ==============================================================================

"""
Configuration Notes:
1. Update all file paths to match your local environment
2. Adjust batch size based on your GPU memory
3. Classification threshold affects the number of SDG assignments
4. Lower thresholds = more SDG assignments but possibly lower precision
5. Higher thresholds = fewer assignments but higher confidence
"""
