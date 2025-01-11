import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from sklearn.preprocessing import LabelEncoder

# Load the datasets
movielens_1m_path = "path_to_movielens_1m.csv"
movielens_32m_path = "path_to_movielens_32m.csv"
imdb_reviews_path = "path_to_imdb_reviews.csv"

# Load and preprocess MovieLens 1M data
movielens_1m = pd.read_csv(movielens_1m_path)

# Encode demographic information
gender_encoder = LabelEncoder()
movielens_1m['gender_encoded'] = gender_encoder.fit_transform(movielens_1m['gender'])

age_groups = {"1": "Under 18", "18": "18-24", "25": "25-34", "35": "35-44", "45": "45-49", "50": "50-55", "56": "56+"}
movielens_1m['age_group'] = movielens_1m['age'].map(age_groups)

# Prepare LightFM Dataset
dataset = Dataset()
dataset.fit(users=movielens_1m['user_id'],
            items=movielens_1m['movie_id'],
            user_features=movielens_1m[['gender_encoded', 'age_group']].values.ravel(),
            item_features=None)  # Extend to include item features if available

(interactions, weights) = dataset.build_interactions(
    (x["user_id"], x["movie_id"]) for x in movielens_1m.itertuples())

# Add user features
user_features = dataset.build_user_features(
    (x["user_id"], [x["gender_encoded"], x["age_group"]]) for x in movielens_1m.itertuples())

# Train the model
model = LightFM(loss='warp')
model.fit(interactions, user_features=user_features, epochs=30, num_threads=2)

# Recommendation function
def recommend_movies(model, dataset, user_id, user_features, n_rec=5):
    """Generate movie recommendations for a user."""
    n_items = dataset.interactions_shape()[1]
    user_ids = np.array([user_id])
    scores = model.predict(user_ids, np.arange(n_items), user_features=user_features)
    top_items = np.argsort(-scores)[:n_rec]

    return top_items

# Example usage
example_user_id = 1
top_movies = recommend_movies(model, dataset, example_user_id, user_features)
print(f"Top movies for user {example_user_id}: {top_movies}")
