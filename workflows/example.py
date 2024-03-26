import typing
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from flytekit import task, workflow

# Define your movie data (for demonstration, you can replace this with your own dataset)
movie_data = {
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    'movie_id': [101, 102, 101, 102, 102, 103, 103, 104, 104, 105],
    'rating': [5, 4, 4, 3, 5, 3, 4, 2, 3, 1]
}
movies_df = pd.DataFrame(movie_data)

@task
def load_data() -> pd.DataFrame:
  """Load movie data."""
  return movies_df

@task
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess movie data if needed."""
    # Pivot the DataFrame to create a user-movie matrix
    pivoted_data = data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    return pivoted_data

@task
def train_svd(data: pd.DataFrame, n_components: int = 2) -> TruncatedSVD:
  """Train Truncated SVD model."""
  svd = TruncatedSVD(n_components=n_components)
  svd.fit(data)
  return svd

@task
def recommend_movies(svd: TruncatedSVD, user_id: int, n_recommendations: int, data: pd.DataFrame) -> typing.List[int]:
  """Recommend movies based on the encountered user_id."""
  user_ratings = data.loc[user_id].dropna()

  # Prepare user-specific data for prediction
  user_ratings_numeric = np.array(user_ratings.values).reshape(1, -1)
  
  # Predict ratings for the user
  predicted_ratings = svd.transform(user_ratings_numeric)
  
  # Calculate similarity between user preferences and movie ratings
  similarity = np.dot(predicted_ratings, svd.components_)
  
  # Get indices of top recommended movies
  top_movie_indices = similarity.argsort(axis=1)[:, -n_recommendations:]

  # Extract movie IDs for recommendations and convert to integers
  recommendations = [int(data.columns[idx]) for idx in top_movie_indices[0]]

  return recommendations

@workflow
def movie_recommendation_workflow() -> typing.List[int]:
    """Workflow to recommend movies."""
    data = load_data()
    preprocessed_data = preprocess_data(data=data)
    svd_model = train_svd(data=preprocessed_data, n_components=5)
    recommendations = recommend_movies(svd=svd_model, user_id=2, data=preprocessed_data, n_recommendations=2)
    return recommendations # type: ignore

if __name__ == "__main__":
    # Execute the workflow
    print(f"Recommended Movies: {movie_recommendation_workflow()}")
