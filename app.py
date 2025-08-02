# Filename: app.py

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load precomputed embeddings and NCO data
@st.cache(allow_output_mutation=True)
def load_data():
    # Sample NCO data with precomputed embeddings should be present in the same folder
    df = pd.read_pickle('nco_sample_with_embeddings.pkl')  
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return df, model

def main():
    st.title("AI-powered NCO Code Search (Prototype)")

    nco_df, model = load_data()

    user_input = st.text_input("Describe the job (any language):")

    if user_input:
        user_emb = model.encode([user_input])
        nco_embs = np.vstack(nco_df['embedding'].values)
        scores = cosine_similarity(user_emb, nco_embs)[0]
        nco_df['score'] = scores
        top_matches = nco_df.sort_values('score', ascending=False).head(5)

        st.write("Top matching occupation codes:")
        st.table(top_matches[['code', 'description', 'score']])

if __name__ == "__main__":
    main()
