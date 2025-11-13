# AI Concerns to SDG Classification

Multi-label classification framework for mapping AI-related concerns from Twitter/X to UN Sustainable Development Goals (SDGs). Implements Zero-Shot, Fine-tuned BERT, and XLNet models to analyze public discourse on AI risks across 17 SDGs.

## 📄 Paper

This repository contains the implementation code for the paper:

**"Mapping AI Concerns to Sustainable Development Goals using Multi-label Classification"**

- **Dataset**: 719,871 tweets from Twitter/X (January 2022 - January 2025)
- **Best Model**: Fine-tuned BERT achieving **96.4% accuracy**
- **Task**: Multi-label classification across all 17 UN Sustainable Development Goals

## 🎯 Key Features

- **Data Collection**: Query-based Twitter/X data collection via API
- **Preprocessing Pipeline**: Robust text cleaning and filtering
- **Multiple Classification Methods**:
  - Fine-tuned BERT (96.4% accuracy)
  - Pre-trained XLNet (94.3% accuracy)
  - Zero-Shot Classification (78.3% accuracy)
- **Comprehensive Analysis**: SDG distribution analysis and visualization
- **Reproducible**: Well-documented code with configurable parameters

## 📊 Research Highlights

### SDG Distribution
The analysis revealed that AI concerns most frequently align with:
- **SDG 9** (Industry, Innovation and Infrastructure): 52.8%
- **SDG 16** (Peace, Justice and Strong Institutions): 11.8%
- **SDG 5** (Gender Equality): 9.4%

### Model Performance Comparison

| Method | Accuracy | F1 Score | Precision | Recall |
|--------|----------|----------|-----------|--------|
| Fine-tuned BERT | 96.4% | 94.7% | 94.3% | 95.4% |
| Pre-trained XLNet | 94.3% | 91.7% | 93.0% | 91.2% |
| Zero-Shot | 78.3% | 68.8% | 72.1% | 69.6% |

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI-Concerns-SDG-Classification.git
cd AI-Concerns-SDG-Classification
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Data Preparation

1. Place your raw data files in the `data/raw/` directory
2. Supported formats:
   - CSV files with columns: `text`, `tweet.created_at`, `tweet.user_id_str`, `tweet.id_str`
   - JSON files with tweet objects

### Configuration

Edit the configuration section in `sdg_classification_pipeline.py`:

```python
# Update these paths for your environment
DATA_DIR = './data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = './results'
PLOTS_DIR = './plots'

# Adjust processing parameters
MIN_WORDS = 5              # Minimum words per tweet
MIN_CHARS = 50             # Minimum characters per tweet
CLASSIFICATION_THRESHOLD = 0.3  # Confidence threshold for SDG classification
BATCH_SIZE = 16            # Batch size for model inference
```

## 📖 Usage

### Running the Full Pipeline

```bash
python sdg_classification_pipeline.py
```

This will execute:
1. Data preprocessing (cleaning, filtering)
2. SDG classification using fine-tuned BERT
3. Results analysis and visualization
4. Summary report generation

### Running Individual Components

#### 1. Data Preprocessing Only

```python
from sdg_classification_pipeline import preprocess_data

df = preprocess_data(
    input_path='./data/raw/tweets_raw.csv',
    output_path='./data/processed/preprocessed_data.csv'
)
```

#### 2. SDG Classification Only

```python
from sdg_classification_pipeline import SDGClassifier
import pandas as pd

# Load preprocessed data
df = pd.read_csv('./data/processed/preprocessed_data.csv')

# Initialize classifier
classifier = SDGClassifier()

# Classify
df_classified = classifier.classify_dataframe(
    df=df,
    text_column='cleaned_text',
    output_path='./results/sdg_classified_results.csv'
)
```

#### 3. Analysis and Visualization Only

```python
from sdg_classification_pipeline import analyze_sdg_distribution, generate_sdg_summary_report
import pandas as pd

# Load classified data
df = pd.read_csv('./results/sdg_classified_results.csv')

# Analyze distribution
analyze_sdg_distribution(df, output_path='./plots/sdg_distribution.png')

# Generate summary report
summary = generate_sdg_summary_report(
    df, 
    output_path='./results/summary_report.json'
)
```

### Using Zero-Shot Classification (Alternative Method)

```python
from sdg_classification_pipeline import ZeroShotSDGClassifier

# Initialize zero-shot classifier
classifier = ZeroShotSDGClassifier()

# Classify a single text
text = "AI systems may perpetuate gender bias in hiring decisions"
predictions = classifier.classify_text(text)
print(predictions)
```

## 📁 Project Structure

```
AI-Concerns-SDG-Classification/
├── data/
│   ├── raw/                    # Raw data files (CSV, JSON)
│   └── processed/              # Preprocessed data
├── results/                    # Classification results
│   └── summary_report.json     # Summary statistics
├── plots/                      # Visualization outputs
│   └── sdg_distribution.png    # SDG distribution plot
├── sdg_classification_pipeline.py  # Main pipeline script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── LICENSE                     # License information
```

## 🔧 Key Functions

### Data Processing
- `combine_csv_files()`: Combine multiple CSV files
- `json_to_csv()`: Convert JSON data to CSV format
- `preprocess_data()`: Complete preprocessing pipeline
- `clean_text()`: Text cleaning (lowercase, remove URLs, mentions, hashtags)
- `filter_short_texts()`: Remove texts below minimum length thresholds

### Classification
- `SDGClassifier`: Fine-tuned BERT-based classification
  - `classify_text()`: Classify single text
  - `classify_batch()`: Batch classification
  - `classify_dataframe()`: Classify entire DataFrame
- `ZeroShotSDGClassifier`: Zero-shot classification using sentence transformers

### Analysis
- `analyze_sdg_distribution()`: Visualize SDG distribution
- `generate_sdg_summary_report()`: Generate comprehensive summary statistics
- `filter_by_query_pattern()`: Filter data by query patterns

## 📊 Output Files

### Classification Results CSV
Contains original text, cleaned text, and SDG predictions:
```csv
text,cleaned_text,SDG_1,SDG_2,...,SDG_17,num_sdgs
"AI bias concerns...",ai bias concerns...,0.0,0.0,...,0.85,1
```

### Summary Report JSON
```json
{
  "total_texts": 367782,
  "sdg_distribution": {
    "SDG_1": {"count": 3690, "percentage": 1.0, "label": "No Poverty"},
    ...
  },
  "top_5_sdgs": [...],
  "multi_label_stats": {
    "avg_sdgs_per_text": 1.23,
    "texts_with_single_sdg": 289456,
    "texts_with_multiple_sdgs": 78326
  }
}
```

## 🎓 Methodology

### Data Collection
- **Source**: Twitter/X API v2
- **Keywords**: "artificial intelligence concerns", "AI concerns", "AI risks", "AI worries", "AI fear"
- **Time Period**: January 2022 - January 2025
- **Initial Dataset**: 908,190 tweets
- **After Preprocessing**: 719,871 unique tweets
- **Classified Texts**: 367,782 tweets (51.1%)

### Preprocessing Steps
1. Convert all text to strings
2. Remove duplicates
3. Convert to lowercase
4. Remove URLs, mentions, and hashtags
5. Filter texts with < 5 words or < 50 characters

### Classification Approach
- **Model**: `sadickam/sdg-classification-bert` from HuggingFace
- **Architecture**: BERT-based multi-label classifier
- **Threshold**: 0.3 (configurable)
- **Multi-label**: Texts can be assigned to multiple SDGs simultaneously

## 📝 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{your_paper_2025,
  title={Mapping AI Concerns to Sustainable Development Goals using Multi-label Classification},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2025}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📧 Contact

**Mohammadreza**
- Institution: Stony Brook University
- Email: your.email@stonybrook.edu
- GitHub: [@yourusername](https://github.com/yourusername)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OSDG Community Dataset for providing labeled SDG data for model evaluation
- HuggingFace for pre-trained models and transformers library
- Twitter/X for API access to collect data
- Stony Brook University for research support

## 📚 References

Key references for the methodology:
- BERT: Devlin et al. (2019) - "BERT: Pre-training of Deep Bidirectional Transformers"
- XLNet: Yang et al. (2019) - "XLNet: Generalized Autoregressive Pretraining"
- OSDG Dataset: Pàmies et al. (2023) - "OSDG Community Dataset"
- SDG Classification: Savci & Das (2024) - "SDG Classification with BERT"

---

**⭐ If you find this repository useful, please consider giving it a star!**
