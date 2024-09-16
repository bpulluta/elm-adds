import glob
import pandas as pd

def get_corpus():
    """Get the corpus of text data with embeddings."""
    
    # Check the JSON files
    corpus_files = sorted(glob.glob('/Users/bpulluta/elm/examples/adds/embed/*.json'))
    if not corpus_files:
        print("No JSON files found in the specified directory.")
        return None
    
    print(f"Found {len(corpus_files)} JSON files.")

    # Read the JSON files
    try:
        corpus = [pd.read_json(fp) for fp in corpus_files]
        corpus = pd.concat(corpus, ignore_index=True)
    except Exception as e:
        print(f"Error reading JSON files: {e}")
        return None
    
    # Check the CSV file
    try:
        meta = pd.read_csv('/Users/bpulluta/elm/examples/adds/meta.csv')
    except FileNotFoundError:
        print("Meta CSV file not found.")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    # Ensure 'id' field is present
    if 'id' not in corpus.columns:
        print("'id' column not found in corpus.")
        return None
    if 'id' not in meta.columns:
        print("'id' column not found in meta.")
        return None

    # Merge dataframes on 'id'
    corpus['id'] = corpus['id'].astype(str)
    meta['id'] = meta['id'].astype(str)
    corpus = corpus.set_index('id')
    meta = meta.set_index('id')

    corpus = corpus.join(meta, on='id', rsuffix='_record', how='left')

    # Create reference column
    try:
        ref = [f"{row['title']} ({row['url']})" for _, row in corpus.iterrows()]
        corpus['ref'] = ref
    except KeyError as e:
        print(f"Key error when creating reference column: {e}")
        return None

    return corpus
