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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from textblob import TextBlob
from typing import Tuple, List, Dict
from tqdm.auto import tqdm
from datetime import datetime

def log_memory_usage():
    process = psutil.Process(os.getpid())
    logger.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

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

# Returns dictionary of model file paths
def get_model_paths():
    save_dir = 'trained_models'
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    return {
        'model_base': f'{save_dir}/model_base_{timestamp}.joblib',
        'model_user': f'{save_dir}/model_user_{timestamp}.joblib',
        'dataset': f'{save_dir}/dataset_{timestamp}.joblib',
        'test_interactions': f'{save_dir}/test_interactions_{timestamp}.joblib'
    }

# Save trained models and associated data
def save_models(models, dataset, test_interactions):
    try:
        model_files = get_model_paths()

        # Save models and data
        joblib.dump(models['model_base'], model_files['model_base'])
        joblib.dump(models['model_user'], model_files['model_user'])
        joblib.dump(dataset, model_files['dataset'])
        joblib.dump(test_interactions, model_files['test_interactions'])
        
        # Save cross-validation metadata if available
        if 'cv_scores' in models:
            metadata = {
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'cv_scores': models['cv_scores'],
                'best_base_fold': models.get('best_base_fold'),
                'best_user_fold': models.get('best_user_fold'),
                'mean_auc': models.get('mean_auc')
            }
            joblib.dump(metadata, f"{os.path.dirname(model_files['model_base'])}/cv_metadata_{metadata['timestamp']}.joblib")
        
        logger.info("Models and metadata saved successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error saving models: {str(e)}")
        return False

# Load most recent trained models
def load_latest_models():
    try:
        save_dir = 'trained_models'
        if not os.path.exists(save_dir):
            return None, None, None
        
        # Find latest timestamp
        files = os.listdir(save_dir)
        if not files:
            logger.warning("No model files found")
            return None, None, None
        
        latest = max(files, key=lambda x: os.path.getctime(os.path.join(save_dir, x)))
        timestamp = latest.split('_')[-1].replace('.joblib','')
        
        # Load models
        models = {
            'model_base': joblib.load(f'{save_dir}/model_base_{timestamp}.joblib'),
            'model_user': joblib.load(f'{save_dir}/model_user_{timestamp}.joblib')
        }
        dataset = joblib.load(f'{save_dir}/dataset_{timestamp}.joblib')
        test_interactions = joblib.load(f'{save_dir}/test_interactions_{timestamp}.joblib')
        
        logger.info(f"Loaded models from timestamp: {timestamp}")
        return models, dataset, test_interactions
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return None, None, None

class DatasetError(Exception):
    """Custom exception for dataset-related errors"""
    pass

# Load and preprocess datasets
def load_and_preprocess_data():
    try:
        logger.info("Loading datasets...")
        # Load MovieLens 32M dataset
        ratings_32m = pd.read_csv('ml-32m/ratings.csv', encoding='iso-8859-1')
        ratings_32m['source'] = '32m'
        movies_32m = pd.read_csv('ml-32m/movies_v2.csv', encoding='iso-8859-1')
        
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
        user_features = pd.get_dummies(users[['gender', 'occupation']], sparse=True)
        
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
        logger.info("User Features Head:\n%s", user_features.head())
        return user_features.set_index(users['userId'])        
        
    except Exception as e:
        logger.error(f"Error creating user features: {str(e)}")
        raise

def create_imdb_user_features(imdb_reviews):
    # Sentiment analysis on IMDb reviews
    imdb_reviews['sentiment'] = imdb_reviews['Review'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Aggregate user features
    imdb_user_features = imdb_reviews.groupby('User').agg({
        'sentiment': ['mean', 'std'],
        'Review': 'count'
    }).reset_index()
    
    imdb_user_features.columns = ['User', 'avg_sentiment', 'sentiment_std', 'review_count']
    
    # Normalize review count
    max_review_count = imdb_user_features['review_count'].max()
    imdb_user_features['normalized_review_count'] = imdb_user_features['review_count'] / (max_review_count if max_review_count > 0 else 1)

    logger.info("IMDb User Features Head:\n%s", imdb_user_features.head())
    
    return imdb_user_features.set_index('User')

# Create item features
def create_item_features(movies, imdb_reviews):
    
    # Log initial size
    logger.info(f"Initial movies shape: {movies.shape}")

    # Genre features
    genre_features = movies['genres'].str.get_dummies(sep='|')
    logger.info(f"Genre features shape: {genre_features.shape}")

    try:
        # Calculate film age
        current_year = datetime.now().year
        movies['release_date'] = pd.to_datetime(movies['release_date'], format='%Y-%m-%d', errors='coerce')
        movies['release_year'] = movies['release_date'].dt.year

        # Store original index
        original_index = movies.index

        # Calculate film age
        movies['film_age'] = current_year - movies['release_year']
        movies['film_age'] = movies['film_age'].fillna(movies['film_age'].median())
        logger.info(f"Movies shape after date processing: {movies.shape}")

        # Validate conversion
        logger.info(f"Release year range: {movies['release_year'].min()} - {movies['release_year'].max()}")
        logger.info(f"Film age range: {movies['film_age'].min()} - {movies['film_age'].max()}")

        if movies['release_year'].isna().any():
            logger.warning(f"Found {movies['release_year'].isna().sum()} movies with missing release years")
            # Fill missing years with median
            median_year = movies['release_year'].median()
            movies['release_year'] = movies['release_year'].fillna(median_year)
            movies['film_age'] = movies['film_age'].fillna(current_year - median_year)
    
    except Exception as e:
        logger.error(f"Error processing release dates: {str(e)}")
        raise
    
    # Normalize film age
    #movies['film_age'] = (movies['film_age'] - movies['film_age'].min()) / (movies['film_age'].max() - movies['film_age'].min())
    
    # Add IMDb review information
    imdb_reviews['sentiment'] = imdb_reviews['Review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    imdb_reviews = imdb_reviews.rename(columns={'imdbId': 'imdb_id'})
    imdb_movie_features = imdb_reviews.groupby('imdb_id').agg({
        'sentiment': ['mean', 'std'],
        'Review': 'count'
    }).reset_index()
    imdb_movie_features.columns = ['imdb_id', 'imdb_avg_sentiment', 'imdb_sentiment_std', 'imdb_review_count']

    # Handle numerical features
    numerical_features = ['vote_average', 'vote_count', 'runtime', 'revenue', 'budget', 'popularity', 'film_age']
    
    # Fill missing values
    for feature in numerical_features:
        movies[feature] = movies[feature].fillna(movies[feature].median())

    # Handle outliers using winsorization
    def winsorize(s, limits=(0.05, 0.95)):
        return s.clip(s.quantile(limits[0]), s.quantile(limits[1]))
    for feature in numerical_features:
        movies[feature] = winsorize(movies[feature])

    # Normalize numerical features
    scaler = StandardScaler()
    movies[numerical_features] = scaler.fit_transform(movies[numerical_features])

    # Log shapes before merge
    logger.info(f"Movies before IMDb merge: {len(movies)}")
    logger.info(f"IMDb features before merge: {len(imdb_movie_features)}")

    # Merge IMDb features with movies
    movies_with_imdb = movies.merge(imdb_movie_features, on='imdb_id', how='left')
    logger.info(f"Movies after merge: {len(movies_with_imdb)}")

    # Fill NaN values for movies without IMDb reviews
    for col in ['imdb_avg_sentiment', 'imdb_sentiment_std', 'imdb_review_count']:
        movies_with_imdb[col] = movies_with_imdb[col].fillna(0)

    # Normalize IMDb review count
    max_review_count = movies_with_imdb['imdb_review_count'].max()
    movies_with_imdb['normalized_imdb_review_count'] = movies_with_imdb['imdb_review_count'] / (max_review_count if max_review_count > 0 else 1)
    
    # Combine all features
    item_features = pd.concat([
        genre_features, 
        movies_with_imdb[numerical_features + ['film_age', 'imdb_avg_sentiment', 'imdb_sentiment_std', 'normalized_imdb_review_count']]
    ], axis=1)
    
    # Fill any remaining NaN values with 0
    item_features = item_features.fillna(0)

    logger.info("Item Features Head:\n%s", item_features.head())

    # Ensure alignment with original movies
    item_features = item_features.reindex(index=movies.index)
    logger.info(f"Final item_features shape: {item_features.shape}")

    return item_features

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

# Train new models or load existing ones
def train_or_load_models(ratings, user_features, item_features, force_retrain=False, n_folds=5):

    save_success = False  # Initialize the variable

    try:
        if not force_retrain:
            logger.info("Attempting to load existing models...")
            models, dataset, test_interactions = load_latest_models()
            if all(model is not None for model in models):
                logger.info("Successfully loaded existing models")
                return models, dataset, test_interactions
        
        logger.info("Training new models...")
        models, dataset, test_interactions = train_models(
            ratings=ratings, 
            user_features=user_features, 
            item_features=item_features,
            n_folds=n_folds
        )
        
        # Save newly trained models
        save_success = save_models(models, dataset, test_interactions)
        if not save_success:
            logger.warning("Failed to save models, but training completed successfully")

        return models, dataset, test_interactions
        
    except Exception as e:
            logger.error(f"Error in train_or_load_models: {str(e)}")
            raise

# Helper function to train a single model with progress tracking
def train_model_with_progress(model, interactions, name, **kwargs):
    logger.info(f"Starting {name} training with interactions shape: {interactions.shape}")
    validate_interactions(interactions)

    model = LightFM(
        loss='warp',
        no_components=64, 
        learning_rate=0.05,
        item_alpha=1e-6,
        user_alpha=1e-6,
        max_sampled=10,
        learning_schedule='adagrad',
        num_threads=4
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
def train_models(ratings, user_features, item_features, n_folds=5):
    try:
        logger.info("Training recommendation models with cross-validation...")

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
            (interactions, weights) = dataset.build_interactions(ratings[['userId', 'movieId', 'rating']].values)
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
            # Convert user features to float32 for consistency
            user_features = user_features.astype(np.float32)
            
            # Create user feature list with explicit dtype handling
            user_feature_list = [
                (uid, {
                    feature: float(value) 
                    for feature, value in row.items() 
                    if pd.notnull(value) and value != 0
                })
                for uid, row in user_features.iterrows()
            ]
            
            # Build user features with explicit dtype
            user_features_matrix = dataset.build_user_features(user_feature_list)
            
            logger.info(f"Built user features with shape: {user_features_matrix.shape}")
            logger.info(f"User features dtype: {user_features_matrix.dtype}")
        except Exception as e:
            logger.error(f"Failed to build user features: {str(e)}")
            raise
        
        # Prepare item features
        logger.info("Building item features...")
        try:
            item_feature_list = [
                (iid, {feature: value for feature, value in row.items() if pd.notnull(value) and value != 0})
                for iid, row in item_features.iterrows()
            ]
            item_features_matrix = dataset.build_item_features(item_feature_list)
            logger.info(f"Built item features with shape: {item_features_matrix.shape}")
        except Exception as e:
            logger.error(f"Failed to build item features: {str(e)}")
            raise
        
        # Initialize models
        base_model_cv = []
        user_model_cv = []

        # Perform k-fold cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_indices = kf.split(range(interactions.shape[0]))

        for fold, (train_idx, test_idx) in enumerate(fold_indices):
            logger.info(f"Training fold {fold+1}/{n_folds}")

            # Convert to csr_matrix format and properly slice the sparse matrix
            train = sparse.csr_matrix((
                interactions.data[train_idx],
                (interactions.row[train_idx], interactions.col[train_idx])
            ), shape=interactions.shape)

            test = sparse.csr_matrix((
                interactions.data[test_idx],
                (interactions.row[test_idx], interactions.col[test_idx])
            ), shape=interactions.shape)

            # Train base model
            base_model = LightFM(
                loss='warp',
                no_components=64,
                learning_rate=0.05,
                item_alpha=1e-6,
                user_alpha=1e-6,
                max_sampled=10,
                learning_schedule='adagrad',
                random_state=42
            )
            
            # Train user model
            user_model = LightFM(
                loss='warp',
                no_components=64,
                learning_rate=0.05,
                item_alpha=1e-6,
                user_alpha=1e-6,
                max_sampled=10,
                learning_schedule='adagrad',
                random_state=42
            )

            # Train models with early stopping
            best_auc = 0
            patience = 3
            no_improvement = 0

            for epoch in range(30):
                base_model.fit_partial(
                    train,
                    item_features=item_features_matrix,
                    epochs=1,
                    num_threads=4
                )
                
                user_model.fit_partial(
                    train,
                    user_features=user_features_matrix,
                    item_features=item_features_matrix,
                    epochs=1,
                    num_threads=4
                )

                # Evaluate on test set
                current_auc = np.mean([
                    auc_score(base_model, test, item_features=item_features_matrix),
                    auc_score(user_model, test, user_features=user_features_matrix, item_features=item_features_matrix)
                ])
                
                if current_auc > best_auc:
                    best_auc = current_auc
                    no_improvement = 0
                else:
                    no_improvement += 1
                
                if no_improvement >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            base_model_cv.append(base_model)
            user_model_cv.append(user_model)

            logger.info(f"Fold {fold+1} AUC: {best_auc:.4f}")

        # Select best models based on validation performance
        best_base_idx = np.argmax([
            auc_score(model, interactions, item_features=item_features_matrix).mean()
            for model in base_model_cv
        ])

        best_user_idx = np.argmax([
            auc_score(model, interactions, user_features=user_features_matrix, item_features=item_features_matrix).mean()
            for model in user_model_cv
        ])

        return {
            'model_base': base_model_cv[best_base_idx],
            'model_user': user_model_cv[best_user_idx]
        }, dataset, interactions
    
    except Exception as e:
        logger.error(f"Error in train_models: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise

def create_new_user_features(age: int, gender: str, favorite_imdb_ids: List[str], movies_df: pd.DataFrame, imdb_reviews: pd.DataFrame):
    try:
        if not isinstance(age, (int, float)) or age < 0 or age > 120:
            raise ValueError(f"Invalid age value: {age}")
        if gender not in ['M', 'F']:
            raise ValueError(f"Invalid gender value: {gender}")
        if not favorite_imdb_ids or len(favorite_imdb_ids) == 0:
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

        # Get genre preferences from favorite movies
        favorite_movies = movies_df[movies_df['imdb_id'].isin(favorite_imdb_ids)]

        # Calculate genre preferences
        genre_columns = [col for col in movies_df.columns if col.startswith('genre_')]
        for col in genre_columns:
            user_features[col] = favorite_movies[col].mean()
        
        for movie in favorite_movies:
            if movie in movies_df['title'].values:
                genres = movies_df[movies_df['title'] == movie]['genres'].iloc[0].split('|')
                for genre in genres:
                    user_features[f'genre_{genre}'] += 1
        
        # Add preferences based on IMDb reviews of favorite movies
        favorite_reviews = imdb_reviews[imdb_reviews['imdb_id'].isin(favorite_imdb_ids)]
        if not favorite_reviews.empty:
            user_features['review_sentiment'] = favorite_reviews['sentiment'].mean()
            
        # Normalize genre features
        genre_columns = [col for col in user_features.columns if col.startswith('genre_')]
        user_features[genre_columns] = user_features[genre_columns] / len(favorite_movies)
        
        logger.info("Successfully created new user features")
        return user_features
        
    except Exception as e:
        logger.error(f"Error creating new user features: {str(e)}")
        raise

def ensemble_predict(user_ids, item_ids, dataset, model_base, model_user, user_features=None):
    """Generate ensemble predictions for given user-item pairs"""
    base_scores = model_base.predict(
        user_ids=user_ids,
        item_ids=item_ids,
        item_features=dataset.build_item_features()
    )
    
    user_scores = model_user.predict(
        user_ids=user_ids,
        item_ids=item_ids,
        user_features=user_features,
        item_features=dataset.build_item_features()
    )
    
    return 0.4 * base_scores + 0.6 * user_scores

def generate_recommendations(user_id, user_features, dataset, movies, model_base, model_user, n=10):
    try:
        if n <= 0:
            raise ValueError(f"Invalid number of recommendations requested: {n}")
            
        logger.info(f"Generating {n} recommendations...")

        item_ids = np.arange(dataset.interactions_shape[1])

        # Get predictions from both models
        base_scores = model_base.predict(
            user_ids=[user_id] * len(item_ids),
            item_ids=item_ids,
            item_features=dataset.build_item_features()
        )

        user_scores = model_user.predict(
            user_ids=[user_id] * len(item_ids),
            item_ids=item_ids,
            user_features=user_features,
            item_features=dataset.build_item_features()
        )
        
        # Combine predictions (0.4 base + 0.6 user model)
        ensemble_scores = 0.4 * base_scores + 0.6 * user_scores

        # Get top N recommendations
        top_items = item_ids[np.argsort(-ensemble_scores)][:n]
        top_scores = ensemble_scores[np.argsort(-ensemble_scores)][:n]

        # Get movie details
        recommendations = []
        for item_id, score in zip(top_items, top_scores):
            movie = movies[movies['movieId'] == item_id].iloc[0]
            recommendations.append({
                'movieId': item_id,
                'title': movie['title'],
                'score': score,
                'genres': movie['genres'],
                'base_score': base_scores[item_id],
                'user_score': user_scores[item_id]
            })

        logger.info("Successfully generated recommendations")
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise

def explain_recommendations(recommendations, movies, dataset, model_base, model_user, user_features=None):
    explanations = []
    
    for rec in recommendations:
        movie_id = rec['movieId']
        movie = movies[movies['movieId'] == movie_id].iloc[0]
        
        explanation = {
            'title': movie['title'],
            'base_contribution': rec['base_score'] * 0.4,
            'user_contribution': rec['user_score'] * 0.6,
            'total_score': rec['score'],
            'genres': movie['genres'],
            'features': {}
        }
        
        # Get feature contributions from both models
        if user_features is not None:
            user_biases = model_user.get_user_representations(user_features)[1]
            explanation['user_preferences'] = {
                feature: bias for feature, bias in zip(dataset.user_features, user_biases)
                if abs(bias) > 0.1
            }
        
        item_features = dataset.build_item_features()
        item_biases = model_base.get_item_representations(item_features)[1]
        explanation['item_features'] = {
            feature: bias for feature, bias in zip(dataset.item_features, item_biases)
            if abs(bias) > 0.1
        }
        
        explanations.append(explanation)
        
    return explanations

# Main function
def main():
    try:
        logger.info("Starting recommendation system...")
        
        # Load and preprocess data
        ratings, movies, users, imdb_reviews = load_and_preprocess_data()
        
        # Create features
        user_features = create_user_features(users)
        #imdb_user_features = create_imdb_user_features(imdb_reviews)
        item_features = create_item_features(movies, imdb_reviews)
        
        # Check for NaN values
        print("NaN values in ratings:", ratings.isna().sum().sum())
        print("NaN values in user_features:", user_features.isna().sum().sum())
        print("NaN values in item_features:", item_features.isna().sum().sum())
        #print("NaN values in imdb_user_features:", imdb_user_features.isna().sum().sum())

        # Before training, validate data
        if not all(isinstance(uid, (int, np.integer)) for uid in ratings['userId']):
            raise ValueError("All user IDs must be integers")
            
        if not all(isinstance(mid, (int, np.integer)) for mid in ratings['movieId']):
            raise ValueError("All movie IDs must be integers")
            
        if not ratings['rating'].between(1, 5).all():
            raise ValueError("All ratings must be between 1 and 5")

        models, dataset, test_interactions = train_or_load_models(
            ratings=ratings,
            user_features=user_features, 
            item_features=item_features,
            force_retrain=True,
            n_folds=5
        )

        model_base = models['model_base']
        model_user = models['model_user']
        
        # Example: Generate recommendations for a new user
        new_user_features = create_new_user_features(
            age=25,
            gender='M',
            favorite_imdb_ids=['0166924', '0068646', '0110912', '0099685', '0172495'], # Mulholland Drive, 'The Godfather', 'Pulp Fiction', 'Godfellas', 'Gladiator'
            movies_df=movies, 
            imdb_reviews=imdb_reviews
        )
        
        # Generate recommendations
        recommendations = generate_recommendations(
            user_id=999999,  # new user ID
            user_features=new_user_features,
            dataset=dataset,
            movies=movies,
            model_base=model_base,
            model_user=model_user,
            n=10
        )

        # Get explanations
        explanations = explain_recommendations(
            recommendations=recommendations,
            movies=movies,
            dataset=dataset,
            model_base=model_base,
            model_user=model_user,
            user_features=new_user_features
        )

        for rec, exp in zip(recommendations, explanations):
            print(f"\nMovie: {rec['title']}")
            print(f"Total Score: {rec['score']:.3f}")
            print(f"Base Model Contribution: {exp['base_contribution']:.3f}")
            print(f"User Model Contribution: {exp['user_contribution']:.3f}")
            print("Genre Match:", exp['genres'])
            if 'user_preferences' in exp:
                print("User Preferences:", exp['user_preferences'])        

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application failed: {str(e)}")
        raise
