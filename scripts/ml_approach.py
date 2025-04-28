# -*- coding: utf-8 -*-
"""Machine Learning Approach for Instagram Caption Generation"""

# Standard library imports
import os
import time
import gc
from collections import defaultdict
from typing import Dict, List, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import cv2
import torch
import joblib
from tqdm import tqdm
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torchvision.models import mobilenet_v2
from torchvision import transforms

# Constants
PROJECT_ROOT = '/content/drive/MyDrive/instagram_post_generator/'
TRAIN_PATH = os.path.join(PROJECT_ROOT, 'Flicker8k_Dataset/train')
TEST_PATH = os.path.join(PROJECT_ROOT, 'Flicker8k_Dataset/test')
VAL_PATH = os.path.join(PROJECT_ROOT, 'Flicker8k_Dataset/val')
TOKENS_PATH = os.path.join(PROJECT_ROOT, 'Flickr8k_text/Flickr8k.token.txt')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'model/ml_description_model.pkl')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'outputs/ml_captions.csv')

# Helper Functions
def clear_memory():
    """Clear memory between operations"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_image_captions(tokens_file: str) -> Dict[str, List[str]]:
    """Load {image_filename: [list of tokenized captions]} from tokens file"""
    captions = defaultdict(list)
    with open(tokens_file) as f:
        for line in f:
            if line.strip():  # Skip empty lines
                parts = line.strip().split('\t')
                if len(parts) == 2:  # Ensure proper format
                    img_file = parts[0].split('#')[0]  # Remove #0, #1 etc.
                    caption = parts[1].lower().split()  # Tokenized and lowercase
                    captions[img_file].append(caption)
    return dict(captions)

def get_image_paths(image_dir: str, captions_dict: Dict[str, List[str]]) -> List[Tuple[str, List[str]]]:
    """
    Get aligned (image_path, captions) pairs for a directory
    
    Args:
        image_dir: Path to directory containing images
        captions_dict: Dictionary mapping filenames to caption lists
        
    Returns:
        List of tuples (image_path, captions)
    """
    return [
        (os.path.join(image_dir, img_file), captions_dict[img_file])
        for img_file in os.listdir(image_dir)
        if img_file.endswith('.jpg') and img_file in captions_dict
    ]

# Main Model Class
class CaptionGenerator:
    def __init__(self, cache_path="./feature_cache", device='cpu', batch_size=8):
        """Initialize the caption generator model"""
        self.model = LogisticRegression(
            max_iter=1000,
            C=0.1,
            solver='saga',
            n_jobs=-1,
            verbose=1
        )
        self.vec = DictVectorizer(sparse=True)
        self.word2idx = {}
        self.idx2word = {}
        self.cache_path = cache_path
        self.device = device
        self.batch_size = batch_size
        self.is_fitted = False
        self.cnn_model = None
        self.transform = None
        os.makedirs(self.cache_path, exist_ok=True)
        print(f"Initialized CaptionGenerator (device: {device})")

    def _load_cnn_model(self):
        """Load CNN model only when needed"""
        if self.cnn_model is None:
            print("Loading CNN model...")
            self.cnn_model = mobilenet_v2(pretrained=True)
            self.cnn_model = torch.nn.Sequential(*list(self.cnn_model.children())[:-1])
            self.cnn_model.eval()
            self.cnn_model.to(self.device)
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
            print("CNN model loaded")

    def extract_image_features(self, image_path: str) -> np.ndarray:
        """Extract image features using CNN"""
        cache_file = os.path.join(self.cache_path, f"{os.path.basename(image_path)}_features.npy")
        if os.path.exists(cache_file):
            return np.load(cache_file)

        self._load_cnn_model()
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.cnn_model(img_tensor).squeeze().cpu().numpy()

            if len(features.shape) == 3:
                features = features.mean(axis=1).mean(axis=1)
            elif len(features.shape) > 1:
                features = features.flatten()[:1280]

            features = np.array(features, dtype=np.float32)
            np.save(cache_file, features)
            return features
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return np.zeros(1280, dtype=np.float32)

    def create_feature_dict(self, prev_word: str, img_feats: np.ndarray) -> Dict:
        """Create feature dictionary for prediction"""
        feature_dict = {'prev_word': prev_word}
        indices = np.linspace(0, len(img_feats)-1, 40, dtype=int)
        for i, idx in enumerate(indices):
            feature_dict[f'img_{i}'] = float(img_feats[idx])
        return feature_dict

    def train(self, train_images: List[str], train_captions: List[List[str]], 
              save_path: str = "model.pkl", max_captions_per_image: int = 1, 
              max_images: int = 1000) -> 'CaptionGenerator':
        """Train the caption generator model"""
        start_time = time.time()

        if max_images and len(train_images) > max_images:
            print(f"Limiting training to {max_images} images")
            indices = np.random.choice(len(train_images), max_images, replace=False)
            train_images = [train_images[i] for i in indices]
            train_captions = [train_captions[i] for i in indices]

        print("Building vocabulary...")
        all_words = set(['<start>', '<end>'])
        for caps in train_captions:
            if isinstance(caps, str):
                caps = [caps]
            for cap in caps[:max_captions_per_image]:
                if isinstance(cap, str):
                    words = cap.split()
                else:
                    words = cap
                all_words.update(words)

        self.word2idx = {w: i for i, w in enumerate(sorted(all_words))}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        print(f"Vocabulary size: {len(self.word2idx)}")

        total_images = len(train_images)
        total_batches = (total_images + self.batch_size - 1) // self.batch_size
        all_features = []
        all_targets = []

        for batch_start in range(0, total_images, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_images)
            batch_num = batch_start//self.batch_size + 1
            print(f"\nBatch {batch_num}/{total_batches} (images {batch_start+1}-{batch_end}/{total_images})")
            clear_memory()

            batch_features = []
            batch_targets = []

            for img_idx in range(batch_start, batch_end):
                img_path = train_images[img_idx]
                caps = train_captions[img_idx]

                if isinstance(caps, str):
                    caps = [caps]
                caps = caps[:max_captions_per_image]

                try:
                    img_feats = self.extract_image_features(img_path)
                    for cap in caps:
                        if isinstance(cap, str):
                            words = cap.split()
                        else:
                            words = cap
                        words = ['<start>'] + words + ['<end>']

                        for i in range(1, len(words)):
                            prev_word = words[i-1]
                            target_word = words[i]
                            if target_word not in self.word2idx:
                                continue

                            feat_dict = self.create_feature_dict(prev_word, img_feats)
                            batch_features.append(feat_dict)
                            batch_targets.append(self.word2idx[target_word])
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

            all_features.extend(batch_features)
            all_targets.extend(batch_targets)

        if all_features:
            try:
                print(f"\nFinal vectorization of {len(all_features)} examples...")
                X_vec = self.vec.fit_transform(all_features)
                y_array = np.array(all_targets)

                print("Training final model...")
                self.model.fit(X_vec, y_array)
                self.is_fitted = True

                if hasattr(self.model, 'coef_') and self.model.coef_ is not None:
                    print(f"Model successfully fitted with shape: {self.model.coef_.shape}")
                else:
                    print("WARNING: Model does not appear to be properly fitted!")

                self.save_model(save_path)
                clear_memory()
            except Exception as e:
                print(f"Error in final training: {e}")
                self.save_model(save_path.replace('.pkl', '_partial.pkl'))

        print(f"\nTraining completed in {(time.time() - start_time)/60:.2f} minutes")
        return self

    def generate_caption(self, image_path: str, beam_width: int = 3, max_length: int = 15) -> str:
        """Generate caption for an image"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")

        img_feats = self.extract_image_features(image_path)
        beam = [{'sequence': ['<start>'], 'score': 0.0}]

        for _ in range(max_length):
            if all(s['sequence'][-1] == '<end>' for s in beam):
                break

            new_beam = []
            for state in beam:
                if state['sequence'][-1] == '<end>':
                    new_beam.append(state)
                    continue

                prev_word = state['sequence'][-1]
                feat_dict = self.create_feature_dict(prev_word, img_feats)

                try:
                    feats = self.vec.transform([feat_dict])
                    log_probs = self.model.predict_log_proba(feats)[0]
                    top_indices = np.argsort(log_probs)[-beam_width:]
                    for idx in top_indices:
                        if idx in self.idx2word:
                            word = self.idx2word[idx]
                            new_score = state['score'] + log_probs[idx]
                            new_sequence = state['sequence'] + [word]
                            new_beam.append({
                                'sequence': new_sequence,
                                'score': new_score
                            })
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    new_beam.append({
                        'sequence': state['sequence'] + ['<end>'],
                        'score': state['score'] - 10
                    })

            beam = sorted(new_beam, key=lambda x: x['score'], reverse=True)[:beam_width]

        best_sequence = max(beam, key=lambda x: x['score'])['sequence']
        caption = [word for word in best_sequence if word not in ['<start>', '<end>']]
        return ' '.join(caption)

    def save_model(self, path: str, verify: bool = False) -> None:
        """Save model to disk"""
        try:
            print(f"Saving model to {path}...")
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

            model_data = {
                'model': self.model,
                'vec': self.vec,
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'is_fitted': True,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }

            joblib.dump(model_data, path)
            print(f"Model saved to {path}")

        except Exception as e:
            print(f"Error saving model: {e}")
            alt_path = f"backup_model_{int(time.time())}.pkl"
            try:
                joblib.dump(model_data, alt_path)
                print(f"Saved backup model to {alt_path}")
            except Exception as backup_err:
                print(f"Failed to save backup model: {backup_err}")

    def load_model(self, path: str) -> bool:
        """Load model from disk"""
        try:
            print(f"Loading model from {path}...")
            model_data = joblib.load(path)

            self.model = model_data['model']
            self.vec = model_data['vec']
            self.word2idx = model_data['word2idx']
            self.idx2word = model_data['idx2word']
            self.is_fitted = model_data.get('is_fitted', False)

            if not hasattr(self.model, 'coef_') or self.model.coef_ is None:
                print("WARNING: Loaded model is not properly fitted!")
                return False

            if not hasattr(self.vec, 'feature_names_') or not self.vec.feature_names_:
                print("WARNING: Loaded vectorizer is not properly fitted!")
                return False

            print("Model loaded successfully and verified as fitted")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def generate_captions(model, test_images):
        """
        Generate captions for a list of images using the provided model.
        
        Args:
            model: Loaded model for caption generation
            test_images (list): List of image paths to process
            
        Returns:
            list: List of dictionaries containing image names and generated captions
        """
        results = []
        for img_path in tqdm(test_images, desc="Generating captions"):
            try:
                caption = model.generate_caption(img_path)
            except Exception as e:
                caption = f"Error: {e}"
            results.append({'image': os.path.basename(img_path), 'caption': caption})
        return results

def main():
    """Main execution function"""
    # Load data
    all_captions = load_image_captions(TOKENS_PATH)
    train_pairs = get_image_paths(TRAIN_PATH, all_captions)
    val_pairs = get_image_paths(VAL_PATH, all_captions)
    test_pairs = get_image_paths(TEST_PATH, all_captions)

    train_images, train_captions = zip(*train_pairs) if train_pairs else ([], [])
    val_images, val_captions = zip(*val_pairs) if val_pairs else ([], [])
    test_images, test_captions = zip(*test_pairs) if test_pairs else ([], [])

    # Initialize and train model
    model = CaptionGenerator(
        cache_path="./feature_cache",
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=24
    )

    model.train(
        train_images=val_images,
        train_captions=val_captions,
        save_path=MODEL_PATH,
        max_captions_per_image=1,
        max_images=len(train_images)
    )

    # Generate captions
    success = model.load_model(MODEL_PATH)
    if not success:
        raise RuntimeError("Model not loaded or not fitted. Retrain or check your model file.")

    results = []
    for img_path in tqdm(test_images, desc="Generating captions"):
        try:
            caption = model.generate_caption(img_path)
        except Exception as e:
            caption = f"Error: {e}"
        results.append({'image': os.path.basename(img_path), 'caption': caption})

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved captions to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
