import traceback
import psutil
import scipy
import numpy as np
import pandas as pd
import joblib
import logging
import os
from scipy import sparse
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from typing import Tuple, List, Dict
from tqdm.auto import tqdm
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recommendation_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DatasetError(Exception):
    """Custom exception for dataset-related errors"""
    pass

def log_memory_usage():
    process = psutil.Process(os.getpid())
    logger.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# Load and preprocess datasets
def load_and_preprocess_data():
    try:
        logger.info("Loading datasets...")
        # Load MovieLens 32M dataset
        ratings_32m = pd.read_csv('ml-32m/ratings.csv', encoding='iso-8859-1')
        ratings_32m['source'] = '32m'
        movies_32m = pd.read_csv('ml-32m/movies.csv', encoding='iso-8859-1')
        
        # Load MovieLens 1M dataset
        ratings_1m = pd.read_csv('ml-1m/ratings.csv', encoding='windows-1252')
        ratings_1m['source'] = '1m'
        users_1m = pd.read_csv('ml-1m/users.csv', encoding='windows-1252')
        
        # Load IMDb reviews
        imdb_reviews = pd.read_csv('imdb/imdb_reviews.csv', encoding='windows-1252')
        
        # Merge datasets
        logger.info("Merging datasets...")
        ratings_combined = pd.concat([ratings_32m, ratings_1m], ignore_index=True)
        # Group by userId and movieId, then aggregate
        ratings = ratings_combined.groupby(['userId', 'movieId']).agg({
            'rating': 'mean',  # Take the average rating
            'timestamp': 'max',  # Take the most recent timestamp
            'source': lambda x: '|'.join(set(x))  # Join the sources
        }).reset_index()
        # # Identify ratings outside the valid range
        # invalid_ratings = ratings[(ratings['rating'] < 1) | (ratings['rating'] > 5)].copy()

        # # Add columns to show original ratings from each source
        # invalid_ratings['32m_rating'] = invalid_ratings.apply(lambda row: ratings_32m[(ratings_32m['userId'] == row['userId']) & (ratings_32m['movieId'] == row['movieId'])]['rating'].values[0] if '32m' in row['source'] else None, axis=1)
        # invalid_ratings['1m_rating'] = invalid_ratings.apply(lambda row: ratings_1m[(ratings_1m['userId'] == row['userId']) & (ratings_1m['movieId'] == row['movieId'])]['rating'].values[0] if '1m' in row['source'] else None, axis=1)

        # # Export invalid ratings to CSV
        # logger.info("Saving invalid ratings to 'invalid_ratings.csv'...")
        # invalid_ratings.to_csv('invalid_ratings.csv', index=False)
        # logger.info(f"Exported {len(invalid_ratings)} invalid ratings to 'invalid_ratings.csv'")

        movies = movies_32m
        users = users_1m
      
        # Handle missing values
        ratings = ratings.dropna()
        movies = movies.dropna()
        users = users.dropna()

        # Add this after loading the ratings dataframe
        logger.info(f"Original number of ratings: {len(ratings)}")

        # Drop ratings outside 1-5 range
        invalid_ratings = ~ratings['rating'].between(1, 5)
        num_dropped = invalid_ratings.sum()
        ratings = ratings[~invalid_ratings]

        logger.info(f"Dropped {num_dropped} ratings outside valid range (1-5)")
        logger.info(f"Remaining number of ratings: {len(ratings)}")
        
        if ratings.empty or movies.empty or users.empty:
            raise DatasetError("One or more datasets are empty after loading")
            
        logger.info(f"Successfully loaded data: {len(ratings)} ratings, {len(movies)} movies, {len(users)} users")
        return ratings, movies, users, imdb_reviews
        
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {str(e)}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty dataset encountered: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during data loading: {str(e)}")
        raise

# Create user features
def create_user_features(users):
    try:
        logger.info("Creating user features...")
        # Demographic features
        user_features = pd.get_dummies(users[['gender', 'occupation']])
        
        # Handle age groups
        age_groups = {
            1: 'Under 18',
            18: '18-24',
            25: '25-34',
            35: '35-44',
            45: '45-49',
            50: '50-55',
            56: '56+'
        }
        user_features = pd.concat([user_features, pd.get_dummies(users['age'].map(age_groups), prefix='age')], axis=1)
        logger.info(f"Created user features with shape: {user_features.shape}")
        return user_features.set_index(users['userId'])  
        
    except Exception as e:
        logger.error(f"Error creating user features: {str(e)}")
        raise

def create_imdb_user_features(imdb_reviews):
    # Sentiment analysis on IMDb reviews
    imdb_reviews['sentiment'] = imdb_reviews['Review'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Aggregate user features
    user_features = imdb_reviews.groupby('User').agg({
        'sentiment': ['mean', 'std'],
        'Review': 'count'
    }).reset_index()
    
    user_features.columns = ['User', 'avg_sentiment', 'sentiment_std', 'review_count']
    
    # Normalize review count
    max_review_count = user_features['review_count'].max()
    user_features['normalized_review_count'] = user_features['review_count'] / max_review_count
    
    return user_features

# Create item features
def create_item_features(movies, imdb_reviews):
    # Genre features
    genre_features = movies['genres'].str.get_dummies(sep='|')

    # Aggregate review sentiment per movie
    imdb_reviews['sentiment'] = imdb_reviews['Review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    movie_sentiments = imdb_reviews.groupby('Movie')['sentiment'].agg(['mean', 'count']).reset_index()
    movie_sentiments.columns = ['title', 'avg_sentiment', 'review_count']
    
    # Merge sentiment data with movies, keeping all movies
    movies_with_sentiments = movies.merge(movie_sentiments, on='title', how='left')
    
    # Fill NaN values for movies without reviews
    movies_with_sentiments['avg_sentiment'] = movies_with_sentiments['avg_sentiment'].fillna(0)
    movies_with_sentiments['review_count'] = movies_with_sentiments['review_count'].fillna(0)
    
    # Normalize review count
    max_review_count = movies_with_sentiments['review_count'].max()
    movies_with_sentiments['normalized_review_count'] = movies_with_sentiments['review_count'] / (max_review_count if max_review_count > 0 else 1)
    
    # Combine all features
    item_features = pd.concat([
        genre_features, 
        movies_with_sentiments[['avg_sentiment', 'normalized_review_count']]
    ], axis=1)
    
    # Ensure all movies are included and set index to movieId
    item_features = item_features.reindex(movies['movieId'])
    
    # Fill any remaining NaN values with 0
    item_features = item_features.fillna(0)

    return item_features.set_index(movies['movieId'])

def get_age_group(age):
    if age < 18:
        return 1
    elif 18 <= age <= 24:
        return 18
    elif 25 <= age <= 34:
        return 25
    elif 35 <= age <= 44:
        return 35
    elif 45 <= age <= 49:
        return 45
    elif 50 <= age <= 55:
        return 50
    else:
        return 56

def validate_interactions(interactions):
    logger.info("Detailed interaction matrix validation:")
    logger.info(f"Shape: {interactions.shape}")
    logger.info(f"Type: {type(interactions)}")
    logger.info(f"Data type of values: {interactions.dtype}")
    logger.info(f"Non-zero elements: {interactions.nnz}")
    logger.info(f"Density: {interactions.nnz / (interactions.shape[0] * interactions.shape[1]):.4%}")
    logger.info(f"Value range: min={interactions.data.min():.2f}, max={interactions.data.max():.2f}")

    # Memory check
    matrix_size = (interactions.data.nbytes + interactions.indptr.nbytes + 
                  interactions.indices.nbytes) / 1024 / 1024
    logger.info(f"Matrix memory usage: {matrix_size:.2f} MB")
    
    if not isinstance(interactions, scipy.sparse.csr_matrix):
        raise ValueError("Interactions must be a CSR matrix")
    if np.isnan(interactions.data).any():
        raise ValueError("Interactions contain NaN values")
    
    return True

def train_model_with_progress(model, interactions, name, **kwargs):
    """Helper function to train a single model with progress tracking"""
    logger.info(f"Starting {name} training with interactions shape: {interactions.shape}")
    validate_interactions(interactions)

    model = LightFM(
        loss='logistic',
        no_components=64, 
        learning_rate=0.05,
        item_alpha=1e-6,
        user_alpha=1e-6,
        max_sampled=10 
    )
    
    try:
        batch_size = interactions.shape[0] // 4
        for epoch in range(30):
            logger.info(f"{name}: Starting epoch {epoch}/30")
            for start_idx in range(0, interactions.shape[0], batch_size):
                end_idx = min(start_idx + batch_size, interactions.shape[0])
                batch = interactions[start_idx:end_idx]

                model.fit_partial(
                    interactions,
                    epochs=1,
                    verbose=True,
                    **kwargs
                )

            if epoch % 5 == 0:
                logger.info(f"{name}: Completed epoch {epoch}/30")
        
        logger.info(f"{name}: Training completed successfully")
        return model
    
    except Exception as e:
        logger.error(f"{name}: Training failed - {str(e)}")
        raise

# Train models
def train_models(ratings, user_features, item_features, imdb_user_features):
    try:
        logger.info("Training recommendation models...")

        # Create LightFM datasets
        user_features = user_features.dropna()
        item_features = item_features.dropna()
        ratings = ratings.dropna()
    
        logger.info(f"Starting model training with {len(ratings)} ratings")
        logger.info(f"Shape of user_features: {user_features.shape}")
        logger.info(f"Shape of item_features: {item_features.shape}")
        
        initial_count = len(ratings)
        ratings = ratings[ratings['rating'].between(1, 5)]
        dropped_count = initial_count - len(ratings)
        logger.info(f"Dropped {dropped_count} ratings outside 1-5 range. Remaining: {len(ratings)}")

        # Create Dataset
        logger.info("Creating LightFM dataset...")
        dataset = Dataset()
        
        all_item_ids = set(ratings['movieId'].unique()) | set(item_features.index)

        logger.info(f"Number of unique users: {len(all_item_ids)}")
        logger.info(f"Number of unique items: {len(ratings['userId'].unique())}")

        # Fit the dataset first
        logger.info("Fitting user and item IDs...")
        dataset.fit(
            ratings['userId'].unique(),
            all_item_ids,
            user_features=user_features.columns,
            item_features=item_features.columns
        )
        
        # Build interactions
        logger.info("Building interactions...")
        try:
            interactions, weights = dataset.build_interactions(ratings[['userId', 'movieId', 'rating']].values)
            logger.info(f"Built interactions with shape: {interactions.shape}")
            logger.info(f"Number of non-zero entries: {interactions.nnz}")
            logger.info(f"Density of interactions matrix: {interactions.nnz / (interactions.shape[0] * interactions.shape[1]):.4%}")
        except Exception as e:
            logger.error(f"Failed to build interactions: {str(e)}")
            raise
        #(interactions, weights) = dataset.build_interactions(ratings[['userId', 'movieId', 'rating']].values)
        
        # Prepare user features
        logger.info("Building user features...")
        try:
            user_feature_list = [
                (uid, {feature: value for feature, value in row.items() if pd.notnull(value)})
                for uid, row in user_features.iterrows()
            ]
            user_features_matrix = dataset.build_user_features(user_feature_list)
            logger.info(f"Built user features with shape: {user_features_matrix.shape}")
        except Exception as e:
            logger.error(f"Failed to build user features: {str(e)}")
            raise
        # user_feature_list = [
        #     (uid, {feature: value for feature, value in row.items() if value != 0})
        #     for uid, row in user_features.iterrows()
        # ]
        # user_features_matrix = dataset.build_user_features(user_feature_list)
        
        # Prepare item features
        logger.info("Building item features...")
        try:
            item_feature_list = [
                (iid, {feature: value for feature, value in row.items() if pd.notnull(value)})
                for iid, row in item_features.iterrows()
            ]
            item_features_matrix = dataset.build_item_features(item_feature_list)
            logger.info(f"Built item features with shape: {item_features_matrix.shape}")
        except Exception as e:
            logger.error(f"Failed to build item features: {str(e)}")
            raise
        # item_feature_list = [
        #     (iid, {feature: value for feature, value in row.items() if value != 0})
        #     for iid, row in item_features.iterrows()
        # ]
        # item_features_matrix = dataset.build_item_features(item_feature_list)
        
        # Split data
        logger.info("Splitting into train/test sets...")
        train_interactions, test_interactions = train_test_split(interactions, test_size=0.2, random_state=42)
        
        # Train models
        models = {
            'model_32m': LightFM(loss='logistic'),
            'model_1m': LightFM(loss='logistic'),
            'model_imdb': LightFM(loss='logistic')
        }

        for name, model in models.items():
            logger.info(f"Training {name}...")
            try:
                if name == 'model_32m':
                    models[name] = train_model_with_progress(
                        model,
                        train_interactions,
                        name
                    )
                elif name == 'model_1m':
                    models[name] = train_model_with_progress(
                        model,
                        train_interactions,
                        name,
                        user_features=user_features_matrix
                    )
                else:  # model_imdb
                    models[name] = train_model_with_progress(
                        model,
                        train_interactions,
                        name,
                        item_features=item_features_matrix
                    )
            except Exception as e:
                logger.error(f"Failed to train {name}: {str(e)}")
                raise

        logger.info("All models trained successfully")
        return models['model_32m'], models['model_1m'], models['model_imdb'], dataset, test_interactions

        # logger.info("Training model_32m...")
        # validate_interactions(train_interactions)
        # model_32m = train_model_with_progress(
        #     LightFM(loss='warp'),
        #     train_interactions,
        #     "model_32m"
        # )
        
        # logger.info("Training model_1m...")
        # model_1m = train_model_with_progress(
        #     LightFM(loss='warp'),
        #     train_interactions,
        #     "model_1m",
        #     user_features=user_features_matrix
        # )
        
        # logger.info("Training model_imdb...")
        # model_imdb = train_model_with_progress(
        #     LightFM(loss='warp'),
        #     train_interactions,
        #     "model_imdb",
        #     item_features=item_features_matrix
        # )

        #logger.info("All models trained successfully")
        #return model_32m, model_1m, model_imdb, dataset, test_interactions
        
    except Exception as e:
        logger.error(f"Error in train_models: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise

def create_new_user_features(age: int, gender: str, favorite_movies: List[str], movies_df: pd.DataFrame, imdb_reviews: pd.DataFrame):
    try:
        if not isinstance(age, (int, float)) or age < 0 or age > 120:
            raise ValueError(f"Invalid age value: {age}")
        if gender not in ['M', 'F']:
            raise ValueError(f"Invalid gender value: {gender}")
        if not favorite_movies or len(favorite_movies) == 0:
            raise ValueError("No favorite movies provided")
            
        logger.info(f"Creating features for new user (age: {age}, gender: {gender})")
        # Create demographic features
        user_features = pd.DataFrame({
            'age': [get_age_group(age)],
            'gender': [gender]
        })
        
        # Handle age groups
        age_groups = {
            1: 'Under 18',
            18: '18-24',
            25: '25-34',
            35: '35-44',
            45: '45-49',
            50: '50-55',
            56: '56+'
        }
        user_features = pd.concat([user_features, pd.get_dummies(user_features['age'].map(age_groups), prefix='age')], axis=1)
        user_features = pd.get_dummies(user_features, columns=['gender'])
        
        # Add favorite movie genres
        all_genres = set(movies_df['genres'].str.split('|', expand=True).values.ravel())
        all_genres.discard(None)  # Remove None if present
        
        for genre in all_genres:
            user_features[f'genre_{genre}'] = 0
        
        for movie in favorite_movies:
            if movie in movies_df['title'].values:
                genres = movies_df[movies_df['title'] == movie]['genres'].iloc[0].split('|')
                for genre in genres:
                    user_features[f'genre_{genre}'] += 1
        
        # Normalize genre features
        genre_columns = [col for col in user_features.columns if col.startswith('genre_')]
        user_features[genre_columns] = user_features[genre_columns] / len(favorite_movies)
        
        # Add average sentiment of favorite movies
        favorite_movie_sentiments = imdb_reviews[imdb_reviews['Movie'].isin(favorite_movies)]['Review'].apply(lambda x: TextBlob(x).sentiment.polarity).mean()
        user_features['avg_sentiment'] = favorite_movie_sentiments
        
        logger.info("Successfully created new user features")
        return user_features
        
    except Exception as e:
        logger.error(f"Error creating new user features: {str(e)}")
        raise

def ensemble_predict(models, dataset, user_features, item_ids):
    predictions = []
    for model in models:
        user_features_matrix = dataset.build_user_features(user_features.to_dict('records'), normalize=False)
        pred = model.predict(0, item_ids, user_features=user_features_matrix)
        predictions.append(pred)
    return np.mean(predictions, axis=0)

def generate_recommendations(models, dataset, user_features, movies_df, n=10):
    try:
        if n <= 0:
            raise ValueError(f"Invalid number of recommendations requested: {n}")
            
        logger.info(f"Generating {n} recommendations...")
        predictions = ensemble_predict(models, dataset, user_features, np.arange(len(movies_df)))
        
        if np.isnan(predictions).any():
            raise ValueError("NaN values in predictions")
            
        top_items = np.arange(len(movies_df))[np.argsort(-predictions)][:n]
        recommendations = movies_df.iloc[top_items]['title'].tolist()
        
        logger.info("Successfully generated recommendations")
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise

def explain_recommendations(models, dataset, user_features, item_id, item_features):
    feature_importances = []
    for model in models:
        user_features_matrix = dataset.build_user_features(user_features.to_dict('records'), normalize=False)
        user_biases, user_factors = model.get_user_representations(user_features_matrix)
        item_biases, item_factors = model.get_item_representations(item_features)
        importance = user_factors[0].dot(item_factors[item_id].T)
        feature_importances.append(importance)
    
    avg_importance = np.mean(feature_importances, axis=0)
    top_features = np.argsort(-avg_importance)[:5]
    return top_features

# Main function
def main():
    try:
        logger.info("Starting recommendation system...")
        
        # Load and preprocess data
        ratings, movies, users, imdb_reviews = load_and_preprocess_data()
        
        # Create features
        user_features = create_user_features(users)
        imdb_user_features = create_imdb_user_features(imdb_reviews)
        item_features = create_item_features(movies, imdb_reviews)
        
        # Check for NaN values
        print("NaN values in ratings:", ratings.isna().sum().sum())
        print("NaN values in user_features:", user_features.isna().sum().sum())
        print("NaN values in item_features:", item_features.isna().sum().sum())
        print("NaN values in imdb_user_features:", imdb_user_features.isna().sum().sum())

        # Before training, validate data
        if not all(isinstance(uid, (int, np.integer)) for uid in ratings['userId']):
            raise ValueError("All user IDs must be integers")
            
        if not all(isinstance(mid, (int, np.integer)) for mid in ratings['movieId']):
            raise ValueError("All movie IDs must be integers")
            
        if not ratings['rating'].between(1, 5).all():
            raise ValueError("All ratings must be between 1 and 5")
        
        # Train models
        model_32m, model_1m, model_imdb, dataset, test_interactions = train_models(ratings, user_features, item_features, imdb_user_features)
        
        # Example: Generate recommendations for a new user
        new_user_age = 30
        new_user_gender = 'M'
        new_user_favorite_movies = ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'Pulp Fiction', 'Forrest Gump']
        
        new_user_features = create_new_user_features(new_user_age, new_user_gender, new_user_favorite_movies, movies, imdb_reviews)
        
        recommendations = generate_recommendations([model_32m, model_1m, model_imdb], dataset, new_user_features, movies)
        logger.info(f"Generated recommendations: {recommendations}")
        print("Recommendations for new user:", recommendations)
        
        # Example: Explain a recommendation
        item_id = movies[movies['title'] == recommendations[0]].index[0]
        top_features = explain_recommendations([model_32m, model_1m, model_imdb], dataset, new_user_features, item_id, item_features)
        print("Top features for recommendation:", top_features)

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application failed: {str(e)}")
        raise


## To integrate this with a GUI, you would:
## Collect the new user's age, gender, and top 5 favorite movies through the GUI.
## Pass this information to the create_new_user_features function.
## Use the resulting user features to generate recommendations with the generate_recommendations function.
## Display the recommendations in the GUI.
## Optionally, use the explain_recommendations function to provide explanations for each recommendation.