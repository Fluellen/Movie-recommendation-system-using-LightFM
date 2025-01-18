import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

# Load and preprocess datasets
def load_datasets():
    # Load MovieLens 32M dataset
    ml_32m_ratings = pd.read_csv('ml-32m/ratings.csv')
    ml_32m_movies = pd.read_csv('ml-32m/movies_v2.csv')
    
    # Load MovieLens 1M dataset
    ml_1m_ratings = pd.read_csv('ml-1m/ratings.csv')
    ml_1m_users = pd.read_csv('ml-1m/users.csv')
    
    # Load IMDb reviews
    imdb_reviews = pd.read_csv('imdb/imdb_reviews.csv')
    
    return ml_32m_ratings, ml_32m_movies, ml_1m_ratings, ml_1m_users, imdb_reviews

# Create user features
def create_user_features(ml_1m_users, imdb_reviews):
    # Demographic features
    user_features = ml_1m_users[['userId', 'gender', 'age']]
    user_features['gender'] = user_features['gender'].map({'M': 0, 'F': 1})
    
    # One-hot encode age groups
    age_dummies = pd.get_dummies(user_features['age'], prefix='age')
    user_features = pd.concat([user_features, age_dummies], axis=1)
    user_features.drop('age', axis=1, inplace=True)
    
    # Merge features
    # user_features = user_features.merge(user_sentiment, on='userId', how='left')
    # user_features['avg_sentiment'] = user_features['avg_sentiment'].fillna(0)
    
    return user_features

# Create item features
def create_item_features(ml_32m_movies):
    # Use CountVectorizer for genres
    genre_vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
    genre_matrix = genre_vectorizer.fit_transform(ml_32m_movies['genres'].fillna(''))
 
    item_features = pd.DataFrame(genre_matrix.toarray(), columns=genre_vectorizer.get_feature_names())
    item_features['movieId'] = ml_32m_movies['movieId']
    
    return item_features

# Create IMDb features
def create_imdb_features(imdb_reviews):

    sia = SentimentIntensityAnalyzer()
    imdb_reviews['sentiment'] = imdb_reviews['Review'].apply(lambda x: sia.polarity_scores(x)['compound'])
    imdb_features = imdb_reviews.groupby('Movie')[['sentiment', 'Rating']].mean().reset_index()

    return imdb_features

# Train models
def train_models(ml_32m_ratings, ml_1m_ratings, user_features, item_features, imdb_features):
    # Collaborative Filtering model (SVD)
    reader = Reader(rating_scale=(1, 5))

    data_32m = Dataset.load_from_df(ml_32m_ratings[['userId', 'movieId', 'rating']], reader)
    trainset_32m, testset_32m = train_test_split(data_32m, test_size=0.2, random_state=42)
    
    svd_model = SVD()
    svd_model.fit(trainset_32m)
    
    sim_options = {'name': 'cosine', 'user_based': True}
    knn_model = KNNBasic(sim_options=sim_options)
    knn_model.fit(trainset_32m)
    
    cb_sim = cosine_similarity(item_features.drop('movieId', axis=1))
    
    return svd_model, knn_model, cb_sim, user_features, item_features, imdb_features

# Function to add a new user
def add_new_user(user_features, age, gender):
    new_user_id = user_features['userId'].max() + 1
    new_user = pd.DataFrame({
        'userId': [new_user_id],
        'gender': [0 if gender == 'M' else 1],
    })

    age_groups = {1: 'Under 18', 18: '18-24', 25: '25-34', 35: '35-44', 45: '45-49', 50: '50-55', 56: '56+'}
    for age_code, age_range in age_groups.items():
        new_user[f'age_{age_code}'] = 1 if age in age_range else 0

    return pd.concat([user_features, new_user], ignore_index=True), new_user_id

# Ensemble prediction
def ensemble_predict_new_user(userId, movie_id, svd_model, knn_model, cb_sim, user_features, item_features, imdb_features):
    # Collaborative Filtering prediction
    # We'll use the global mean rating as we don't have user history
    cf_pred = svd_model.trainset.global_mean
    # cf_pred = svd_model.predict(uid, iid).est
    
    # Demographic-based prediction, we'll use demographic similarity
    user_vec = user_features[user_features['userId'] == user_id].iloc[0]
    similar_users = knn_model.get_neighbors(user_id, k=10)
    if similar_users:
        demo_pred = np.mean([knn_model.trainset.global_mean + knn_model.bu[u] for u in similar_users])
    else:
        demo_pred = knn_model.trainset.global_mean
    # demo_pred = knn_model.predict(uid, iid).est
    
    # Content-based prediction
    item_idx = item_features[item_features['movieId'] == movie_id].index[0]
    similar_items = cb_sim[item_idx].argsort()[::-1][1:11]  # Top 10 similar items
    cb_pred = item_features.iloc[similar_items]['sentiment'].mean()
    
    imdb_pred = imdb_features[imdb_features['Movie'] == movie_id]['Rating'].mean() if movie_id in imdb_features['Movie'].values else cf_pred

    # Ensemble prediction (simple average)
    ensemble_pred = (cf_pred + demo_pred + cb_pred + imdb_pred) / 4
    
    return ensemble_pred

# Generate recommendations for new user
def generate_recommendations_new_user(user_id, favorite_movies, svd_model, knn_model, cb_sim, user_features, item_features, imdb_features, n=10):
    # Get all movies except the user's favorites
    all_movies = set(item_features['movieId']) - set(favorite_movies)

    recommendations = []
    for movie_id in all_movies:
        pred = ensemble_predict_new_user(user_id, movie_id, svd_model, knn_model, cb_sim, user_features, item_features)
        recommendations.append((movie_id, pred))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:n]

# Explain recommendations
def explain_recommendations(recommendations, item_features, imdb_features, ml_32m_movies):
    explanations = []
    for movie_id, pred in recommendations:
        movie_info = item_features[item_features['movieId'] == movie_id].iloc[0]
        movie_title = ml_32m_movies[ml_32m_movies['movieId'] == movie_id]['title'].values[0]
        genres = movie_info.index[movie_info > 0].tolist()
        
        imdb_info = imdb_features[imdb_features['Movie'] == movie_id]
        imdb_rating = imdb_info['Rating'].values[0] if not imdb_info.empty else "N/A"
        imdb_sentiment = imdb_info['sentiment'].values[0] if not imdb_info.empty else "N/A"
        
        explanation = f"Movie: {movie_title}\n"
        explanation += f"Predicted rating: {pred:.2f}\n"
        explanation += f"Genres: {', '.join(genres)}\n"
        explanation += f"IMDb rating: {imdb_rating}\n"
        explanation += f"IMDb sentiment: {imdb_sentiment:.2f}\n"
        explanations.append(explanation)
    
    return explanations

# Main function
def main():
    # Load and preprocess data
    ml_32m_ratings, ml_32m_movies, ml_1m_ratings, ml_1m_users, imdb_reviews = load_datasets()
    
    # Create features
    user_features = create_user_features(ml_1m_users)
    item_features = create_item_features(ml_32m_movies)
    imdb_features = create_imdb_features(imdb_reviews)
    
    # Train models
    svd_model, knn_model, cb_sim, user_features, item_features, imdb_features = train_models(
        ml_32m_ratings, ml_1m_ratings, user_features, item_features, imdb_features)
    
    # Example usage
    #user_id = 1
    favorite_movies = [1, 2, 3, 4, 5] # Example movie IDs
    age = 30 # This will be mapped to the '25-34' age group
    gender = 'M'
    
    user_features, new_user_id = add_new_user(user_features, age, gender)
    
    recommendations = generate_recommendations_new_user(
        new_user_id, favorite_movies, svd_model, knn_model, cb_sim, user_features, item_features, imdb_features)
    
    explanations = explain_recommendations(recommendations, item_features, imdb_features, ml_32m_movies)
    
    print(f"Top 10 Recommendations for new user (ID: {new_user_id}):")
    for explanation in explanations:
        print(explanation)
        print("---")

if __name__ == "__main__":
    main()