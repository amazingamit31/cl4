# Import libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Explanation:
    """ 
            Sequential: A linear stack of layers to build the model.
            Embedding: Converts integer-encoded words into dense vectors (e.g., "cat" → [0.2, -0.5, ...]).
            LSTM: Layer to process sequential data with memory cells and gates.
            Dense: Fully connected layer for classification.
            imdb: Preloaded dataset of movie reviews labeled as positive (1) or negative (0).
            sequence: Utilities for padding sequences to a fixed length. 
    """

# Load dataset
vocab_size = 5000  # Use top 5,000 frequent words
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

    """
            Explanation:
            vocab_size=5000: Restrict vocabulary to the 5,000 most frequent words (reduces noise from rare words).
            imdb.load_data(): Loads the IMDB dataset preprocessed into integer sequences.
            x_train/x_test: Lists of reviews, where each word is replaced by its integer index.
            y_train/y_test: Labels (0 or 1).
    """

# Pad sequences to fixed length (400 words)
max_words = 400
x_train = sequence.pad_sequences(x_train, maxlen=max_words)
x_test = sequence.pad_sequences(x_test, maxlen=max_words)
    """
            Explanation:
            max_words=400: Truncate/pad all reviews to 400 words.
            Shorter reviews are padded with zeros (e.g., [0, 0, ..., 12, 42]).
            Longer reviews are truncated to 400 words.
            Why? Neural networks require fixed-length inputs for batch processing.
    """

# Build LSTM model
model = Sequential(name="LSTM_Sentiment_Analysis")
model.add(Embedding(vocab_size, 32, input_length=max_words))  # Convert word indices to 32D vectors
model.add(LSTM(128, activation='tanh', return_sequences=False))  # 128 LSTM units
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    """
            Step 1: Embedding Layer:
            vocab_size: 5,000 unique words.
            32: Embedding dimension (each word is a 32-dimensional vector).
            input_length=max_words: Each input sequence has 400 words.
            Purpose: Converts sparse integer-encoded words into dense vectors that capture semantic meaning (e.g., "good" and "great" are closer in vector space).

            Step 2: LSTM Layer:
            128: Number of LSTM units (dimensionality of the hidden state).
            activation='tanh': Hyperbolic tangent activation for gate updates.
            return_sequences=False: Return only the final output (not all timesteps).
            Purpose: Processes the sequence word-by-word, updating its hidden state to capture context.

            Step 3: Dense Layer:
            1: Single neuron for binary classification (positive/negative).
            activation='sigmoid': Squashes output to [0, 1] (probability of positive sentiment).

    """

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    """
            Explanation:
            loss='binary_crossentropy': Standard loss for binary classification.
            optimizer='adam': Adaptive learning rate optimizer (efficient for RNNs).
            metrics=['accuracy']: Track accuracy during training.
    """

# Train model
history = model.fit(x_train, y_train, 
                   batch_size=64, 
                   epochs=5, 
                   validation_split=0.2)
    """
            Explanation:
            batch_size=64: Update weights after every 64 samples (balance speed/memory).
            epochs=5: Train for 5 full passes over the training data.
            validation_split=0.2: Use 20% of training data for validation (monitor overfitting).
            Output: Training logs show loss/accuracy for training and validation sets.
    """

# Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
    """
            Explanation:
            evaluate(): Computes loss and accuracy on unseen test data.
            test_acc: Accuracy reflects how well the model generalizes to new reviews.
            Typical Output: ~80-88% accuracy depending on hyperparameters.
    """