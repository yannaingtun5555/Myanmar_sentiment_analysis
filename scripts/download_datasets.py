from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("LOADING MYANMAR SENTIMENT DATASET")
print("=" * 60)

# Load the dataset
print("\n📥 Loading dataset from Hugging Face...")
try:
    dataset = load_dataset("chuuhtetnaing/myanmar-social-media-sentiment-analysis-dataset")
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print(f"❌ Error loading: {e}")
    print("\nTrying with specific split...")
    dataset = load_dataset("chuuhtetnaing/myanmar-social-media-sentiment-analysis-dataset", split="train")
    dataset = {"train": dataset}

# Check structure
print(f"\n📊 Dataset structure: {dataset.keys()}")

# Get the train split
if "train" in dataset:
    df = pd.DataFrame(dataset["train"])
else:
    df = pd.DataFrame(dataset)

print(f"\n📈 Dataset size: {len(df)} rows")
print(f"\n📋 Columns: {df.columns.tolist()}")

# Show first few rows
print("\n🔍 First 5 rows:")
print(df.head())

# Check sentiment distribution
print("\n📊 Sentiment Distribution:")
sentiment_counts = df['Sentiment'].value_counts()
for sentiment, count in sentiment_counts.items():
    print(f"   {sentiment}: {count} ({count/len(df)*100:.1f}%)")

# Check for Myanmar text (Text-MM column)
print("\n🇲🇲 Myanmar Text Sample (from 'Text-MM' column):")
for i in range(min(3, len(df))):
    print(f"   {i+1}. {df.iloc[i]['Text-MM'][:100]}...")

# Save to CSV for easy access
df.to_csv('../data/raw/myanmar_sentiment_full.csv', index=False, encoding='utf-8-sig')
print("\n💾 Saved full dataset to: data/raw/myanmar_sentiment_full.csv")

# Create a simplified version (just text and sentiment)
simple_df = df[['Text-MM', 'Sentiment']].copy()
simple_df.columns = ['text', 'sentiment']
simple_df.to_csv('../data/raw/myanmar_sentiment_simple.csv', index=False, encoding='utf-8-sig')
print("💾 Saved simplified version to: data/raw/myanmar_sentiment_simple.csv")

print("\n✅ Done! Ready for training.")


