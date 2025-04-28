import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import cohere
import re
from joblib import load
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize Cohere client
cohere_api = os.getenv("COHERE_API_KEY")
ch = cohere.Client(api_key=cohere_api)

# Instagram caption template
ig_template = """ 
You are an instagram post captions generator. You will be given a description of the content and you will generate a caption for the post. 
The caption should be engaging and relevant to the content. Be sure to include relevant hashtags and emojis. Be creative; just not excessively so.
The caption should be in the first person in English, as if the user is speaking. No more than 20 words.
If the content is unclear, generate a generic caption that is still relevant to the content, by fixating more on the mood of the content.
"""

# Global configuration
EMOTION_COLUMNS = ['is_anger', 'is_disgust', 'is_fear', 
                  'is_joy', 'is_sadness', 'is_surprise']
THRESHOLD = 0.5

# Load models
@st.cache_resource
def load_models():
    try:
        pca_loaded = load('/Users/ruhwang/Desktop/AI/spring2025_courses/aipi540-dl/ig_post_generator/models/pca_model.joblib')
        lgbm_model = load('/Users/ruhwang/Desktop/AI/spring2025_courses/aipi540-dl/ig_post_generator/models/lgbm.joblib')
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return pca_loaded, lgbm_model, embedding_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

pca_loaded, lgbm_model, embedding_model = load_models()

def clean_text(text):
    """Cleans raw text input for processing"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.!?,]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def embeddings_to_dataframe(embeddings, prefix='embedding_'):
    """Convert numpy array of embeddings to pandas DataFrame"""
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
    return pd.DataFrame(
        embeddings,
        columns=[f"{prefix}{i}" for i in range(embeddings.shape[1])]
    )

def get_embeddings(texts):
    """Convert list of texts to embeddings"""
    if isinstance(texts, str):
        texts = [texts]
    embeddings = embedding_model.encode(texts, convert_to_tensor=False)
    return embeddings_to_dataframe(embeddings)

def apply_pca(embeddings_df):
    """Apply PCA transformation to embeddings DataFrame"""
    if pca_loaded is None:
        return embeddings_df.iloc[:, :100] if embeddings_df.shape[1] > 100 else embeddings_df
    pca_results = pca_loaded.transform(embeddings_df.values)
    return embeddings_to_dataframe(pca_results, prefix='pca_')

def predict_emotions(text):
    """Main prediction function"""
    cleaned_text = clean_text(text)
    
    # Initialize default results in case models aren't loaded
    result = {
        'emotions': [],
        'probabilities': {},
        'dominant_emotion': 'neutral',
        'embeddings_dim': 'N/A',
        'pca_dim': 'N/A'
    }
    
    try:
        # Get embeddings as DataFrame
        embeddings_df = get_embeddings(cleaned_text)
        result['embeddings_dim'] = embeddings_df.shape[1]
        
        # Apply PCA
        reduced_embeddings_df = apply_pca(embeddings_df)
        result['pca_dim'] = reduced_embeddings_df.shape[1]
        
        # Predict probabilities
        if lgbm_model is not None:
            probs = np.array([est.predict_proba(reduced_embeddings_df)[:, 1] 
                            for est in lgbm_model.estimators_]).T
        else:
            probs = np.random.dirichlet(np.ones(6), size=1)[0]
        
        # Convert to probability dictionary
        prob_dict = dict(zip(EMOTION_COLUMNS, probs[0] if len(probs.shape) > 1 else probs))
        
        # Apply threshold
        binary_pred = [1 if p >= THRESHOLD else 0 for p in prob_dict.values()]
        emotions = [col.replace('is_', '') 
                  for col, val in zip(EMOTION_COLUMNS, binary_pred) 
                  if val == 1] or ['neutral']
        
        result.update({
            'emotions': emotions,
            'probabilities': prob_dict,
            'dominant_emotion': max(prob_dict.items(), key=lambda x: x[1])[0].replace('is_', '')
        })
        
    except Exception as e:
        st.error(f"Error during emotion prediction: {e}")
    
    return result

def make_caption(query: str, emotion: str) -> str:
    """Generate caption using Cohere API"""
    prompt = ig_template + f"\n\nCurrent emotion: {emotion}\nContent description: {query}"
    response = ch.chat(
        model="command-r-plus",
        message=prompt,
        temperature=0.5
    )
    return response.text

def create_emotion_barplot(prob_dict):
    """Create a bar plot of emotion probabilities"""
    emotions = [e.replace('is_', '') for e in prob_dict.keys()]
    probabilities = list(prob_dict.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0', '#118AB2', '#073B4C']
    bars = ax.bar(emotions, probabilities, color=colors)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    ax.set_ylabel('Probability')
    ax.set_title('Emotion Probability Distribution')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig

def main():
    st.title("Instagram Content Analyzer")
    
    tab1, tab2 = st.tabs(["Caption Generator", "ML Emotion Analysis"])
    
    with tab1:
        st.header("Caption Generator")
        content_description = st.text_area(
            "Describe your post content:",
            placeholder="e.g., 'Enjoying a beautiful day at the beach with friends'",
            height=100
        )
        
        if content_description and st.button("Generate Caption"):
            with st.spinner("Analyzing content and generating caption..."):
                # Predict emotions
                emotion_result = predict_emotions(content_description)
                
                # Generate caption
                caption = make_caption(content_description, emotion_result['dominant_emotion'])
                
                # Display results
                st.subheader("Generated Caption")
                st.write(caption)
                
                st.subheader("Detected Mood")
                st.write(f"Dominant emotion: {emotion_result['dominant_emotion'].title()}")
                
                # Show emotion plot
                st.pyplot(create_emotion_barplot(emotion_result['probabilities']))
                
                # Store results
                st.session_state['caption_result'] = emotion_result
    
    with tab2:
        st.header("ML Emotion Analysis")
        
        if 'caption_result' in st.session_state:
            result = st.session_state['caption_result']
            
            st.subheader("Raw Model Output")
            prob_df = pd.DataFrame.from_dict(
                result['probabilities'], 
                orient='index', 
                columns=['Probability']
            )
            prob_df.index = prob_df.index.str.replace('is_', '').str.title()
            st.dataframe(prob_df.style.format("{:.2%}").background_gradient(cmap='Blues'))
            
            st.subheader("Binary Predictions")
            binary_preds = [1 if p >= THRESHOLD else 0 for p in result['probabilities'].values()]
            binary_df = pd.DataFrame({
                'Emotion': prob_df.index,
                'Predicted': ['Yes' if b else 'No' for b in binary_preds]
            })
            st.dataframe(binary_df)

if __name__ == "__main__":
    main()