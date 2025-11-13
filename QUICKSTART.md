# Quick Start Guide

This guide will help you get started with the AI-Concerns-SDG-Classification pipeline in just a few minutes.

## ⚡ Quick Setup (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Create the data directory structure:

```bash
mkdir -p data/raw data/processed results plots
```

Place your Twitter/X data in `data/raw/`. The file should be a CSV with at least a `text` column.

Example CSV format:
```csv
text,tweet.created_at,tweet.user_id_str,tweet.id_str
"AI bias in hiring is concerning...",2023-04-15,123456789,987654321
"Worried about AI privacy issues...",2023-04-16,123456790,987654322
```

### 3. Run the Pipeline

```bash
python sdg_classification_pipeline.py
```

That's it! The pipeline will:
- Preprocess your data
- Classify tweets to SDGs
- Generate visualizations and reports

Results will be saved in:
- `results/` - Classification results and summary report
- `plots/` - SDG distribution visualization

## 🎯 Common Use Cases

### Use Case 1: Classify New Tweets

```python
from sdg_classification_pipeline import SDGClassifier

# Initialize classifier
classifier = SDGClassifier()

# Classify a single tweet
text = "AI systems may amplify existing gender biases in workplace decisions"
predictions = classifier.classify_text(text)
print(predictions)
# Output: {'SDG_5': 0.87, 'SDG_8': 0.42}
```

### Use Case 2: Batch Process a CSV File

```python
from sdg_classification_pipeline import preprocess_data, SDGClassifier
import pandas as pd

# Step 1: Preprocess
df = preprocess_data(
    input_path='./data/raw/my_tweets.csv',
    output_path='./data/processed/my_tweets_clean.csv'
)

# Step 2: Classify
classifier = SDGClassifier()
df_classified = classifier.classify_dataframe(
    df=df,
    output_path='./results/my_results.csv'
)

print(f"Classified {len(df_classified)} tweets")
```

### Use Case 3: Analyze Existing Results

```python
from sdg_classification_pipeline import analyze_sdg_distribution, generate_sdg_summary_report
import pandas as pd

# Load classified data
df = pd.read_csv('./results/sdg_classified_results.csv')

# Visualize distribution
analyze_sdg_distribution(df, output_path='./plots/my_distribution.png')

# Generate report
summary = generate_sdg_summary_report(df, output_path='./results/my_summary.json')
```

### Use Case 4: Customize Configuration

```python
# Create a custom configuration
from sdg_classification_pipeline import SDGClassifier

# Initialize with custom threshold
classifier = SDGClassifier(threshold=0.5)  # Higher threshold = more confident predictions

# Or modify global settings
import sdg_classification_pipeline as pipeline
pipeline.MIN_WORDS = 10  # Require at least 10 words
pipeline.MIN_CHARS = 100  # Require at least 100 characters
pipeline.BATCH_SIZE = 32  # Use larger batches if you have more GPU memory
```

## 📊 Understanding the Output

### Classification Results CSV

After running the pipeline, you'll get a CSV file with these columns:

| Column | Description |
|--------|-------------|
| `text` | Original tweet text |
| `cleaned_text` | Preprocessed text |
| `SDG_1` to `SDG_17` | Confidence scores for each SDG (0-1) |
| `num_sdgs` | Number of SDGs assigned to this tweet |

Example row:
```csv
text,cleaned_text,SDG_1,SDG_5,SDG_8,SDG_9,...,num_sdgs
"AI bias in hiring...","ai bias hiring...",0.0,0.87,0.42,0.0,...,2
```

This tweet is classified as:
- **SDG 5 (Gender Equality)**: 87% confidence
- **SDG 8 (Decent Work)**: 42% confidence
- Total: 2 SDGs assigned

### Summary Report JSON

The summary report contains:

```json
{
  "total_texts": 367782,
  "sdg_distribution": {
    "SDG_1": {
      "count": 3690,
      "percentage": 1.0,
      "label": "No Poverty"
    },
    ...
  },
  "top_5_sdgs": [
    {"sdg": 9, "count": 194236, "label": "Industry, Innovation and Infrastructure"}
  ],
  "multi_label_stats": {
    "avg_sdgs_per_text": 1.23,
    "texts_with_single_sdg": 289456,
    "texts_with_multiple_sdgs": 78326
  }
}
```

## 🔧 Troubleshooting

### Issue: Out of Memory Error

**Solution**: Reduce batch size

```python
from sdg_classification_pipeline import SDGClassifier

classifier = SDGClassifier()
# Reduce batch size in classification
df_classified = classifier.classify_batch(texts, batch_size=8)  # Default is 16
```

### Issue: Classification Takes Too Long

**Solution**: Enable GPU acceleration

```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

If GPU is not available, consider:
1. Using Google Colab with GPU runtime
2. Processing data in smaller chunks
3. Reducing the dataset size for testing

### Issue: Too Many/Too Few SDG Assignments

**Solution**: Adjust classification threshold

```python
# For fewer, more confident assignments
classifier = SDGClassifier(threshold=0.5)  # Default is 0.3

# For more assignments with lower confidence
classifier = SDGClassifier(threshold=0.2)
```

### Issue: Import Errors

**Solution**: Reinstall requirements

```bash
pip install --upgrade -r requirements.txt
```

## 💡 Tips for Better Results

### 1. **Data Quality Matters**
- Ensure your tweets are in English (or adjust preprocessing for other languages)
- Remove spam, promotional content, and bot-generated tweets
- Keep tweets that actually discuss AI concerns

### 2. **Preprocessing Parameters**
- Increase `MIN_WORDS` and `MIN_CHARS` for better quality (but fewer tweets)
- Decrease them to include shorter tweets (but possibly noisier data)

### 3. **Classification Threshold**
- **0.2-0.3**: More inclusive, captures weak signals (recommended for exploration)
- **0.4-0.5**: Balanced approach (recommended for production)
- **0.6-1.0**: Very conservative, high confidence only

### 4. **Multi-label Considerations**
- Most AI concern tweets relate to multiple SDGs
- Average is ~1.2 SDGs per tweet in our study
- Don't be surprised if many tweets map to SDG 9 (Innovation) and SDG 16 (Institutions)

## 📚 Next Steps

1. **Read the full paper** for methodology details
2. **Explore the code** in `sdg_classification_pipeline.py`
3. **Customize for your needs** using `config_template.py`
4. **Join the discussion** by opening GitHub issues
5. **Contribute** by submitting pull requests

## 🆘 Getting Help

If you encounter issues:

1. Check this Quick Start Guide
2. Review the main [README.md](README.md)
3. Look at code comments in `sdg_classification_pipeline.py`
4. Open a GitHub issue with:
   - Error message
   - Python version
   - GPU/CPU info
   - Sample data (if possible)

## 📞 Contact

For questions or collaboration:
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/AI-Concerns-SDG-Classification/issues)
- **Email**: your.email@stonybrook.edu

---

Happy classifying! 🎉
