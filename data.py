from sentence_transformers import SentenceTransformer
import pandas as pd

# Load your NCO sample CSV
df = pd.read_csv('nco_sample.csv')

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
df['embedding'] = df['description'].apply(lambda x: model.encode(x))

df.to_pickle('nco_sample_with_embeddings.pkl')
print("Saved embeddings to nco_sample_with_embeddings.pkl")
