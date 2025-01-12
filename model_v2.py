import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

# Load and preprocess datasets
def load_datasets():
    # Load MovieLens 32M dataset
    ml_32m_ratings = pd.read_csv('ml-32m/ratings.csv')
    ml_32m_movies = pd.read_csv('ml-32m/movies.csv')
    
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
    
    # Sentiment features
    sia = SentimentIntensityAnalyzer()
    imdb_reviews['sentiment'] = imdb_reviews['Review'].apply(lambda x: sia.polarity_scores(x)['compound'])
    user_sentiment = imdb_reviews.groupby('User')['sentiment'].mean().reset_index()
    user_sentiment.columns = ['userId', 'avg_sentiment']
    
    # Merge features
    # user_features = user_features.merge(user_sentiment, on='userId', how='left')
    # user_features['avg_sentiment'] = user_features['avg_sentiment'].fillna(0)
    
    return user_features

# Create item features
def create_item_features(ml_32m_movies, imdb_reviews):
    # Genre features
    ml_32m_movies['genres'] = ml_32m_movies['genres'].fillna('')
    tfidf = TfidfVectorizer()
    genre_matrix = tfidf.fit_transform(ml_32m_movies['genres'])
    
    # Sentiment features
    imdb_reviews['sentiment'] = imdb_reviews['Review'].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound'])
    item_sentiment = imdb_reviews.groupby('Movie')['sentiment'].mean().reset_index()
    
    # Merge features
    item_features = pd.DataFrame(genre_matrix.toarray(), columns=tfidf.get_feature_names())
    item_features['movieId'] = ml_32m_movies['movieId']
    item_features = item_features.merge(item_sentiment, left_on='movieId', right_on='movieId', how='left')
    item_features['sentiment'] = item_features['sentiment'].fillna(0)
    
    return item_features

# Train models
def train_models(ml_32m_ratings, ml_1m_ratings, user_features, item_features):
    # Collaborative Filtering model (SVD)
    reader = Reader(rating_scale=(1, 5))

    data_32m = Dataset.load_from_df(ml_32m_ratings[['userId', 'movieId', 'rating']], reader)
    trainset_32m, testset_32m = train_test_split(data_32m, test_size=0.2, random_state=42)
    
    svd_model = SVD()
    svd_model.fit(trainset_32m)
    
    # Demographic-based model
    data_1m = Dataset.load_from_df(ml_1m_ratings[['userId', 'movieId', 'rating']], reader)
    trainset_1m, testset_1m = train_test_split(data_1m, test_size=0.2, random_state=42)
    
    sim_options = {'name': 'cosine', 'user_based': True}
    knn_model = KNNBasic(sim_options=sim_options)
    knn_model.fit(trainset_1m)
    
    # Content-based model
    cb_sim = cosine_similarity(item_features.drop(['movieId', 'Movie', 'sentiment'], axis=1))
    
    return svd_model, knn_model, cb_sim, testset_32m, testset_1m

# Function to add a new user
def add_new_user(user_features, age, gender):
    new_user_id = user_features['userId'].max() + 1
    new_user = pd.DataFrame({
        'userId': [new_user_id],
        'gender': [0 if gender == 'M' else 1],
        'age': [age]
    })
    return pd.concat([user_features, new_user], ignore_index=True), new_user_id

# Ensemble prediction
def ensemble_predict_new_user(uid, iid, svd_model, knn_model, cb_sim, user_features, item_features):
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
    item_idx = item_features[item_features['movieId'] == iid].index[0]
    similar_items = cb_sim[item_idx].argsort()[::-1][1:11]  # Top 10 similar items
    cb_pred = item_features.iloc[similar_items]['sentiment'].mean()
    
    # Ensemble prediction (simple average)
    ensemble_pred = (cf_pred + demo_pred + cb_pred) / 3
    
    return ensemble_pred

# Generate recommendations for new user
def generate_recommendations_new_user(user_id, favorite_movies, age, gender, svd_model, knn_model, cb_sim, user_features, item_features, n=10):
    # Get all movies except the user's favorites
    all_movies = set(item_features['movieId']) - set(favorite_movies)

    recommendations = []
    for movie_id in all_movies:
        pred = ensemble_predict_new_user(user_id, movie_id, svd_model, knn_model, cb_sim, user_features, item_features)
        recommendations.append((movie_id, pred))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:n]

# Explain recommendations
def explain_recommendations(recommendations, item_features):
    explanations = []
    for movie_id, pred in recommendations:
        movie_info = item_features[item_features['movieId'] == movie_id].iloc[0]
        genres = movie_info.index[movie_info > 0].tolist()
        sentiment = movie_info['sentiment']
        
        explanation = f"Movie ID: {movie_id}\n"
        explanation += f"Predicted rating: {pred:.2f}\n"
        explanation += f"Genres: {', '.join(genres)}\n"
        explanation += f"Average sentiment: {sentiment:.2f}\n"
        explanations.append(explanation)
    
    return explanations

# Main function
def main():
    # Load and preprocess data
    ml_32m_ratings, ml_32m_movies, ml_1m_ratings, ml_1m_users, imdb_reviews = load_datasets()
    
    # Create features
    user_features = create_user_features(ml_1m_users)
    item_features = create_item_features(ml_32m_movies, imdb_reviews)
    
    # Train models
    svd_model, knn_model, cb_sim, user_features, item_features = train_models(ml_32m_ratings, ml_1m_ratings, user_features, item_features)
    
    # Example usage
    #user_id = 1
    favorite_movies = [1, 2, 3, 4, 5] # Example movie IDs
    age = 25
    gender = 'M'
    
    recommendations = generate_recommendations_new_user(new_user_id, favorite_movies, svd_model, knn_model, cb_sim, user_features, item_features)
    explanations = explain_recommendations(recommendations, item_features)
    
    print(f"Top 10 Recommendations for new user:")
    for movie_id, pred_rating in recommendations:
        movie_title = ml_32m_movies[ml_32m_movies['movieId'] == movie_id]['title'].values[0]
        print(f"Movie: {movie_title}, Predicted Rating: {pred_rating:.2f}")

if __name__ == "__main__":
    main()