#### Problem Statement
The average young adult spends anywhere between 3-4 hours on social media daily, causing significant screen time. AI can potentially make posting faster 
with auto-generated captions based on input images, thus saving some time for users over time. 

Disclaimer: I decided to go with an approach where I sparate different components of the projects for naive approach, deep learning, and machine learning separately. 
I realize this may not fit the original intent of the course requirements strictly, yet it is for my particular project. In the following, I will breakdown how I made the project with 
these different approaches.

#### Dataset 
Dataset for deep learning: Flickr 8K, publicly available, no privacy concerns, low on bias. 
Images and textual descriptions matching the image IDs. Some other text files exist (such as tokens and text files with just image IDs), but I did not use them anyway. 
I generated my own embedding matrix from the huggingface tokenizer. 

Dataset for machine learning: 1 go2emotion from Kaggle that contains raw text, a emotions label, plus other miscellaneous columns that did not make it to model building.

#### WorkFlow
Deep Learning: A RESNET 50 model to extract the features from the images, word2vec model to get the tokenizer —> embeddings from the textual descriptions, then a LSTM model to generate the next-word-prediction of the captions
LLM: (Cohere command-r-plus) pass the caption to the LLM to generate a coherent caption, with relevant emojis and hashtags!
ML model: use the embedding matrix from the deep learning model, apply PCA to reduce dimensions, and then pass the data into a Light BGM model for predict the mood profile (probability mapping —> plotting)

#### Deep Learning
The goal is to generate descriptions from images. 
My image captioning app combines deep learning for both image understanding and natural language generation. This model extract features from images and describes what it sees in natural language by learning embedding features from the descriptions.

How It Works
The model works in three main stages, similar to how a human might describe an image:

1. Processing Image and Extracting Features
First, I used a pre-trained ResNet neural network (the same kind used in modern computer vision) to encode the image into deep-learning accepted format. This network has already learned to recognize objects, scenes, and visual patterns from a large collection of images.

The ResNet extracts a large set of 2,048 visual features from each image, which are then compressed down to 256 key features. This is to distill the most important visual elements that need to be described.

2. Processing Language
The second part of my model handles language. It uses an LSTM (Long Short-Term Memory) network, which excels at understanding sequence-like sentences.

When generating a caption, the model starts with a partial sentence and needs to predict the next word. The LSTM examines the words already in the caption, embeds them into a 256-dimensional space (similar to how modern language models understand words), and processes them to understand the context of what's been said so far.

3. Connecting Vision and Language
The integration occurs when these two systems combine. The model takes:
(A). What it sees (the 256 image features)
(B). What it's said so far (the 256 language features)
These are added together and processed through additional neural network layers that decide which word should come next in the caption. This process repeats word by word until the model generates a complete caption.

Training Process
I trained the model by exposing it to 8000+ images paired with human-written captions. For each training example, the model had to predict the next word in the caption given the image and the words so far. Over time, it learned to generate increasingly accurate and natural-sounding descriptions.

The model has several techniques to improve its learning:

1. Dropout layers to prevent memorization of the training data
2. Adam optimizer for efficient learning
3. Early stopping to prevent overfitting
4. Learning rate scheduling to fine-tune the training process

Evaluation:

Cross-entropy is used to evaluate the model. For more details, visit the notebooks/colab_resnet_lstm.ipynb for demonstrating the training process and train-evaluation curves. 
I also demoed 3 different images as potential usage materials. 

#### Machine Learning (1)
The ML approach to geenrate captions is a Maximum Entropy Markov Model (MEMM) for generating image captions. 
This model combines image features extracted from a CNN with sequential text prediction using a logistic regression classifier. 
It treats caption generation as a sequence of word prediction tasks, where each word depends on the previous word and the image features.

The main features passed into the logistic regression model: 

CNN Feature Extractor: Uses MobileNetV2 (pretrained on ImageNet) to extract visual features from images (train image files)
DictVectorizer: Transforms feature dictionaries into sparse matrices for the classifier.
LogisticRegression Classifier: Predicts the next word in a caption given the previous word and image features.
Beam Search Decoder: Generates captions by maintaining multiple candidate sequences.

Utilization of a Markov process where: 

(1) The state is represented by the current word in the caption.
(2) Transitions between states (words) are conditioned on both the previous word and image features.
(3) The probability of each transition is modeled using logistic regression.

Image Features

Feature Extraction: Images are processed through MobileNetV2 to extract 1280-dimensional feature vectors.
Dimensionality Reduction: Features are reduced to 40 dimensions to improve efficiency.
Caching: Features are cached to disk to avoid redundant computation.

Text Processing

Vocabulary Building: A vocabulary is constructed from all words in the training captions.
Special Tokens: <start> and <end> tokens are added to mark the beginning and end of captions.
Word Indexing: Words are mapped to indices for the classifier.

Training

The idea is that for each word in the description, create feature dictionary containing the previous word and image features. 
Set the target as the current word. Vectorize features using DictVectorizer. Train the logistic regression model to predict the next word.

For each word transition, a feature dictionary is created with the previous word and the 40 image features. 
Images are processed in batches to manage memory usage. The logistic regression model predicts the probability distribution over all possible next words.
The model is saved periodically during training to prevent data loss.

The nice thing about this approach is that it is relatively simple, fairly efficient, and low on resource constraints. I was training in a Google Colab environment with a GPU and 
it finished in just over a couple hours. Limitation would be that the approach will lack a good amount of context and flexibility, as it is limited to just the training data I have. 

#### Machine Learning (2) 

The 2nd machine learning component aims to give a mood profile (breakdown of emotional style) to allow users to see how their caption sounds like. 
The system provides a more detailed understanding of the way the caption comes across to other users to potentially assist with engagement goals. 

A special note is that the original dataset does not have many useful features, so I created embeddings out of the text and used PCA to reduce the dimensionality of the feature space.
I also filtered down the range of emotions to 6 for more legibility for the readers during the demo. 

The chosen model is a MultiOutputClassifier built on LightGBM (lightweight and efficient training). I did hyperparameter tuning to make sure I can get the model to perform better.
The multi-output classifier is to wrap the lightGBM to give me a list of probabilities so that I can map them to emotions. 

Evaluation: 

I generated classification reports, but focusing more on f1-score because I want a balance between precision and recall. 

#### Naive Approach

The naive approach is to take the captions generated by the deep learning model and then pass to a LLM (with some prompt engineering to guide the response) for simpler retrieval.

Evaluation:
I cannot think of worthwhile metrics for this. As this is mostly a generative AI model with outputs, the most I can do is manually check if the outputs make sense. 

#### Future Work 
A larger training dataset with more diverse images.
Explore more retraining to improve model performance. Some ways to improve the deep learning model in particular could include different architectures, more experimentation with learning rates,
and potentially attention mechanisms to make the system overall more robust. As you can see from the notebooks, the next-word prediction model does not produce exactly coherent sentences. 





