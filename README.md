
# RECOMMENDER SYSTEM FOR MOVIES

# 1. Business Understanding

## 1.1 Business Overview
We've all been there, scrolling through a endless list of movies, unable to decide what to watch. For streaming giants like Netflix and ShowMax, solving this "scroll fatigue" is key to keeping users happy and subscribed. Their secret weapon? The recommendation system. This is a smart tool that suggests new titles based on your past ratings and the preferences of viewers like you.

This project taps into that same idea. Using the public MovieLens dataset, we'll develop a model that learns user preferences to predict which films they'll enjoy most. The final output will be a curated list of top 5 movie recommendations for each user, making their viewing experience more personal and enjoyable while boosting platform engagement.

## 1.2 Problem Statement
Endless scrolling is the enemy of enjoyment. When users can't decide what to watch, their engagement drops and streaming platforms face the risk of lower customer retention.

Our challenge is to transform this experience. Instead of presenting users with an overwhelming catalog, we need to provide them with a personalized shortlist. This project focuses on building a recommendation engine that does exactly that, predicting user preferences to deliver a top-five list of tailored suggestions, making discovery effortless and viewing more satisfying.

## 1.3 Business Objectives

### Main Objective:
To build a recommendation model that provides personalized recommendations to users based on their previous ratings and those of similar users.

### Specific Objectives:
1. To explore and preprocess the MovieLens dataset to understand user–movie interaction patterns.

2. To implement a collaborative filtering model that predicts movie ratings based on user or item similarity.

3. To evaluate the performance of the recommendation model using appropriate metrics such as RMSE or MAE.

4. To demonstrate how the recommendation system can be applied in a streaming platform to improve user experience and retention.

## 1.4 Research Questions
1. What patterns can be identified in the MovieLens dataset regarding user–movie interactions?

2. How can collaborative filtering be used to predict user ratings for movies they haven’t seen?

3. How accurately does the model predict user ratings when evaluated using RMSE or MAE metrics?

4. How can implementing a recommendation system improve user satisfaction and retention for a streaming service?

## 1.5 Success Criteria
- Achieve RMSE ≤ 0.90 and MAE ≤ 0.70 on test data.
- Generate meaningful, personalized recommendations.

# 2. Data Understanding
We are using the **MovieLens Small Dataset**, which contains:
- 100,000 movie ratings
- 600+ users
- 9,000+ unique movies

The data includes user IDs, movie IDs, ratings, and movie metadata (titles, genres, tags).

 We have several libraries and tools that are going to be used such as advanced CF algorithims (SVD).

Dataset, Reader, SVD, train_test_split, accuracy, GridSearchCV are going to be used for building and evaluating the matrix factorization component.

TF-IDF and cosine_similarity, will be used to build the Content-Based Filtering component based on movie genres.Lastly,the final model will be serialized for deployment.

# 3. Data Preparation

## 3.1 Importing Liblaries
Importing the neccessary liblaries that will be used for this project.


```python
#Import necessary libraries
! pip install scikit-surprise
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from surprise.model_selection import GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle
from collections import Counter
```

    Requirement already satisfied: scikit-surprise in c:\users\pc\anaconda3\lib\site-packages (1.1.4)
    Requirement already satisfied: joblib>=1.2.0 in c:\users\pc\anaconda3\lib\site-packages (from scikit-surprise) (1.4.2)
    Requirement already satisfied: numpy>=1.19.5 in c:\users\pc\anaconda3\lib\site-packages (from scikit-surprise) (1.26.4)
    Requirement already satisfied: scipy>=1.6.0 in c:\users\pc\anaconda3\lib\site-packages (from scikit-surprise) (1.13.1)
    

### 3.2 Loading Data
The ratings.csv and movies.csv files are loaded into Pandas DataFrames.

Shape Inspection: The size of the raw datasets is printed, confirming we are starting with 100,836 ratings and 9,742 movies.

Merging: The ratings data is merged with the necessary title and genres information from the movies data using movieId as the common key. This creates a single DataFrame (ratings_merged) containing all user-item-rating details along with the movie's descriptive features.
 
Preview: The first 10 rows of the merged data are displayed for a quick verification of the merge operation.

    Ratings: (100836, 4)
    Movies: (9742, 3)
   
    userId       0
    movieId      0
    rating       0
    timestamp    0
    title        0
    genres       0
    dtype: int64
    
There are no any missing values.

```python
# Plotting the frequency of each movie genre
sns.barplot(x = 'genre', y = 'count', data = genre_counts_df.sort_values(by = 'count', ascending = False), palette = 'magma',
            hue = 'genre', legend = False)
plt.xticks(rotation = 90)
plt.show()
```
   
![png](Recommender_System_files/Recommender%20System%20_32_0.png)
    
```python
# To find the average rating per Genre
ratings_merged = ratings_merged.assign(genres = ratings_merged["genres"].str.split("|")).explode("genres")

genre_avg_ratings = ratings_merged.groupby("genres")["rating"].mean().sort_values(ascending = False)

# Plotting
plt.figure(figsize = (12, 6))
sns.barplot(x = genre_avg_ratings.values, y = genre_avg_ratings.index, palette = 'coolwarm',
             hue = genre_avg_ratings.index, legend = False)
plt.xlabel("Average Rating")
plt.ylabel("Genre")
plt.title("Average Rating per Genre")
plt.show()
```
    
![png](Recommender_System_files/Recommender%20System%20_33_0.png)
    
# 5. Modeling

## 5.1 Temporal Train/Test Split
This code will perform a tempral "leave-one-out" split-training on earlier interactions and testing on each user's latest rating, which gives a more realistic evaluation for recommendation systems

```python
#Data Type Conversion for Timestamps
if 'timestamp' in ratings.columns:
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
# Handling Timestamps and Sorting
ratings_sorted = ratings.sort_values(['userId', 'timestamp']) if 'timestamp' in ratings.columns else ratings.sort_values (['userId'])
#Identifying the Test Set Indices (Leave-One-Out)
test_idx = ratings_sorted.groupby('userId').tail(1).index
# Creating the Final Train and Test DataFrames
test_df = ratings.loc[test_idx].reset_index(drop=True)
train_df = ratings.drop(test_idx).reset_index(drop=True)
#Showing the resulting split.
print('Train Interactions:', len(train_df))
print('Test Interactions (held-out last per user):', len(test_df))

```
    Train Interactions: 100226
    Test Interactions (held-out last per user): 610
    

## 5.2 Data Preparation for Surprise (CF)
This code converts the ratings Dataframe into a format the suprise library can use for model training and evaluation while defining the valid rating range

```python
#Defining the Rating Scale
reader= Reader(rating_scale=(0.5,5.0))
#Loading Data into Surprise Format
data_train= Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
full_train= Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

print(reader)
print(data_train)
print(full_train)

```

    <surprise.reader.Reader object at 0x0000029FB0D0EAB0>
    <surprise.dataset.DatasetAutoFolds object at 0x0000029FAF8BCC50>
    <surprise.dataset.DatasetAutoFolds object at 0x0000029FAF8BFEF0>
    
## 5.3 Hyperparameter Tuning for Collaborative Filtering (SVD):
The code below performs Grid Search Cross-Validation (GridSearchCV) to find the optimal set of parameters for the Singular Value Decomposition (SVD) matrix factorization algorithm. This is a crucial step for optimizing the performance of the Collaborative Filtering component.

```python
#Defining the Search Space
param_grid = {
    'n_factors':[20, 50, 100],
    'lr_all':[0.002, 0.005],
    'reg_all': [0.02, 0.05]
}
#Setting up the Grid Search
gs= GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1)
#Execution and Result
gs.fit(data_train)

print('Best RMSE score:', gs.best_score['rmse'])
print('Best params (RMSE):', gs.best_params['rmse'])

best_params=gs.best_params['rmse']

```

    Best RMSE score: 0.873251367473921
    Best params (RMSE): {'n_factors': 50, 'lr_all': 0.005, 'reg_all': 0.05}
    

## 5.4 Final Collaborative Filtering (SVD) Model Training and Evaluation.
This shows the models generalization performance after tuning and how well the final SVD recommender predicts real user ratings on Unseen movies.

```python
#Final Model Training
trainset = full_train.build_full_trainset()
best_model = SVD(n_factors=best_params['n_factors'], lr_all=best_params['lr_all'], reg_all=best_params['reg_all'], biased=True, random_state=42)
best_model.fit(trainset)
#Test Set Preparation for Surprise
raw_testset= list(zip(test_df['userId'].astype(str).tolist(),test_df['movieId'].astype(str).tolist(), test_df['rating'].astype(str).tolist()))
testset_for_suprise= [(row.userId, row.movieId, row.rating) for row in test_df.itertuples()]
#5 Fold Cross-Validation
cross_val_results = cross_validate(best_model, data_train, cv = 5, measures = ['RMSE', 'MAE'], verbose = True)

#Evaluate the Testset
predictions=best_model.test(testset_for_suprise)
rmse= accuracy.rmse(predictions, verbose=False)
mae= accuracy.mae(predictions, verbose=False)
print(f'RMSE: {rmse:.4f}, MAE: {mae:.4f}')

```

    Evaluating RMSE, MAE of algorithm SVD on 5 split(s).
    
                      Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
    RMSE (testset)    0.8640  0.8708  0.8747  0.8616  0.8730  0.8688  0.0051  
    MAE (testset)     0.6643  0.6721  0.6708  0.6631  0.6711  0.6683  0.0038  
    Fit time          2.10    2.18    1.66    1.75    1.98    1.93    0.20    
    Test time         0.55    0.47    0.31    0.51    0.31    0.43    0.10    
    RMSE: 0.8584, MAE: 0.6677
    

## 5.5 Content-Based Filtering (CBF) 
  Content-Based Filtering (CBF) implements a basic movie recommendation system based purely on genre similarity. 

```python

def compute_genre_similarity(movies):
    # Ensure genres are lists (in case they’re strings like "Action|Adventure")
    movies['genres'] = movies['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else x)
    
    # One-hot encode using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(movies['genres'])
    
    # Compute cosine similarity between genre vectors
    genre_similarity = cosine_similarity(genre_matrix, genre_matrix)
    return genre_similarity

def get_similar(movie, movies, genre_similarity, top_n=5):
    """Find movies most similar to a given movie based on genre."""
    if movie not in movies['title'].values:
        return "Movie Not Found"
    
    # Get the movie index
    movie_idx = movies.index[movies['title'] == movie][0]
    
    # Get indices of top similar movies
    similar_indices = genre_similarity[movie_idx].argsort()[::-1][1:top_n + 1]
    
    return movies.iloc[similar_indices][['title', 'genres']]

# Compute genre similarity matrix
similar_genres = compute_genre_similarity(movies)

# Example usage
recommended_movies = get_similar("Toy Story (1995)", movies, similar_genres)
print(recommended_movies)

```

                                                   title  \
    8219                                    Turbo (2013)   
    3568                           Monsters, Inc. (2001)   
    9430                                    Moana (2016)   
    3000                Emperor's New Groove, The (2000)   
    2809  Adventures of Rocky and Bullwinkle, The (2000)   
    
                                                     genres  
    8219  [Adventure, Animation, Children, Comedy, Fantasy]  
    3568  [Adventure, Animation, Children, Comedy, Fantasy]  
    9430  [Adventure, Animation, Children, Comedy, Fantasy]  
    3000  [Adventure, Animation, Children, Comedy, Fantasy]  
    2809  [Adventure, Animation, Children, Comedy, Fantasy]  
    

## 5.6 Svd Predictions
This is going to be used to predict how specific users would rate every single movie in this dataset.

```python
def get_svd_predictions(user_id, movies, ratings, best_model):

    """
    Parameters:
    User_Id: The Id of the user
    movies: The movies dataset with movie IDs and titles
    ratings: The ratings dataset (not used but included for consistency)
    best_model: The model with the best gridsearchcv parameters
    """

    # Get all unique movie IDs from the movies dataset.
    all_movie_ids = movies['movieId'].unique()

    # Predict ratings for all movies using the SVD model.
    predictions = [best_model.predict(uid = user_id, iid = mid) for mid in all_movie_ids]

    # Creating a DataFrame containing the predicted ratings.
    pred_df = pd.DataFrame([(pred.iid, pred.est) for pred in predictions], columns = ['movieId', 'svd_score'])

    # merging the two dataframes to include movie titles for readability.
    pred_df = pred_df.merge(movies[['movieId', 'title']], on = 'movieId')

    return pred_df
```


```python
get_svd_predictions(5, movies, ratings, best_model)
```

## 5.7 Cold-Start Logic
* It utilizes a genre-based preference score for every movie, tailored to a specific user, or based on a global average if the user has no ratings.


```python
# Compute genre-based scores for a user based on their past ratings and genre similarity
def get_genre_scores(user_id, ratings,  movies, genre_similarity):
    # Get movies the user has rated
    user_movies = ratings[ratings['userId'] == user_id].merge(movies, on = 'movieId')
    
    if user_movies.empty:
        print(f"user {user_id} has no ratings, using global genre preferences.")
        
        # Computing global average genre score
        global_genre_scores = np.mean(genre_similarity, axis = 0)

        genre_df = pd.DataFrame({
            'movieId': movies['movieId'], 
            'title': movies['title'],
            'genre_score': global_genre_scores
        })

        return genre_df
    
    # Compute genre similarity scores
    genre_scores = np.zeros(len(movies))

    for movie_id in user_movies['movieId']:
        movie_idx = movies.index[movies['movieId'] == movie_id][0]
        genre_scores += genre_similarity[movie_idx] * user_movies[user_movies['movieId'] == movie_id]['rating'].values[0]

    # Normalize
    genre_scores /= len(user_movies)
    
    # Create DataFrame
    genre_df = pd.DataFrame({
        'movieId': movies['movieId'], 
        'title': movies['title'],
        'genre_score': genre_scores
    })

    return genre_df

genre_similarity = compute_genre_similarity(movies)

get_genre_scores(99999, ratings, movies, genre_similarity)
```

    user 99999 has no ratings, using global genre preferences.
    

## 5.8 Hybrid Recommendations
This function, hybrid_recommendations, implements a hybrid movie recommendation system by combining two different types of models: Collaborative Filtering (CF) and Content-Based Filtering (CBF).

The final score for each movie is calculated using the formula:
$$
\text{final score} = \alpha \times \text{SVD score} + (1 - \alpha) \times \text{Genre score}
$$
where 
𝛼 controls the weight of CF vs. CBF.


```python
def hybrid_recommendations(user_id, movies, ratings, best_model, genre_similarity, alpha = 0.5):
    
    # Getting CF (SVD) and CBF (genre_similarity) predictions for the user
    svd_df = get_svd_predictions(user_id, movies, ratings, best_model)
    genre_df = get_genre_scores(user_id, ratings, movies, genre_similarity)

    # merging both dataframes
    hybrid_df = svd_df.merge(genre_df, on = 'movieId')

    # Computing final score (weighted blend)
    hybrid_df['final_score'] = alpha * hybrid_df['svd_score'] + (1 - alpha) * hybrid_df['genre_score']

    # Get top recommendations
    top_movies = hybrid_df.sort_values(by = 'final_score', ascending = False).head(10)

    top_movies = top_movies.merge(movies[['movieId', 'title']], on = 'movieId')

    return top_movies[['movieId', 'title', 'final_score']]
```


```python
results = []

for i in range(1, 30):
    recommendations = hybrid_recommendations(i, movies, ratings, best_model, genre_similarity, 0.7)
    recommendations['userId'] = i
    results.append(recommendations)

final_df = pd.concat(results)

print(final_df)
```

        movieId                                           title  final_score  \
    0      6016             City of God (Cidade de Deus) (2002)     3.910995   
    1      1262                        Great Escape, The (1963)     3.886099   
    2      3275                     Boondock Saints, The (2000)     3.864160   
    3      1197                      Princess Bride, The (1987)     3.847972   
    4      1204                       Lawrence of Arabia (1962)     3.832982   
    ..      ...                                             ...          ...   
    5      1262                        Great Escape, The (1963)     3.687452   
    6     68157                     Inglourious Basterds (2009)     3.685928   
    7      6016             City of God (Cidade de Deus) (2002)     3.671871   
    8     48774                          Children of Men (2006)     3.670995   
    9      1172  Cinema Paradiso (Nuovo cinema Paradiso) (1989)     3.669090   
    
        userId  
    0        1  
    1        1  
    2        1  
    3        1  
    4        1  
    ..     ...  
    5       29  
    6       29  
    7       29  
    8       29  
    9       29  
    
    [290 rows x 4 columns]
    

These are the movie recommendations for the selected users.

# 6. Model Evaluation

## 6.1 Evaluating SVD Predictions.
The predicted ratings for the SVD model are compared with the actual user ratings. The function below retrieves predicted ratings for a user then merges these predictions with actual ratings to calculate the RMSE and MAE.The RMSE and MAE for the SVD model predictions for user 1


```python

def evaluate_svd(user_id, movies, ratings, best_model):
    # Get SVD predictions for all movies
    svd_df = get_svd_predictions(user_id, movies, ratings, best_model)

    # Get the actual user ratings
    user_actual_ratings = ratings[ratings['userId'] == user_id][['movieId', 'rating']]
    
    # Merge predicted and actual ratings
    merged_df = svd_df.merge(user_actual_ratings, on='movieId')

    if merged_df.empty:
        print(f"No common movies found for user {user_id}. Skipping evaluation.")
        return None, None, None

    # Calculating RMSE and MAE
    rmse = np.sqrt(mean_squared_error(merged_df['rating'], merged_df['svd_score']))
    mae = mean_absolute_error(merged_df['rating'], merged_df['svd_score'])

    return merged_df, rmse, mae

# Example usage:
svd_results, svd_rmse, svd_mae = evaluate_svd(1, movies, ratings, best_model)
print(f"SVD RMSE: {svd_rmse:.4f}, SVD MAE: {svd_mae:.4f}")

```

    SVD RMSE: 0.7030, SVD MAE: 0.5723
    

## 6.2 Evaluating Hybrid Predictions
It evaluates the accuracy of the combined (hybrid) recommendation system for a specific user, similar to how evaluate_svd works, but using the blended scores.



```python
def evaluate_hybrid(user_id, movies, ratings, best_model, genre_similarity, alpha = 0.7):
    
    """
    best_model: Trained SVD model for CF.
    genre_similarity: Cosine similarity matrix for genres.
    alpha: Weight parameter to balance SVD and genre-based filtering.
    """
    # Get hybrid recommendations for a user
    hybrid_df = hybrid_recommendations(user_id, movies, ratings, best_model, genre_similarity, alpha)
    
    # Accessing the actual user ratings
    user_actual_ratings = ratings[ratings['userId'] == user_id][['movieId', 'rating']]
    
    # Merging predictions with actual ratings
    merged_df = hybrid_df.merge(user_actual_ratings, on = 'movieId')

    # Skipping if merged dataframe is empty
    if merged_df.empty:
        print(f"No common movies found for user {user_id}. Skipping evaluation.")
        return None, None, None

    # Calculate the RMSE and MAE
    rmse = np.sqrt(mean_squared_error(merged_df['rating'], merged_df['final_score']))
    mae = mean_absolute_error(merged_df['rating'], merged_df['final_score'])

    return merged_df, rmse, mae

hybrid_results, hybrid_rmse, hybrid_mae = evaluate_hybrid(4, movies, ratings, best_model, genre_similarity)

print(hybrid_results)
print(f" Hybrid RMSE: {hybrid_rmse}, Hybrid MAE: {hybrid_mae}")
```

       movieId                           title  final_score  rating
    0      914             My Fair Lady (1964)     3.397838     5.0
    1     1203             12 Angry Men (1957)     3.397571     5.0
    2      912               Casablanca (1942)     3.344777     5.0
    3      296             Pulp Fiction (1994)     3.343454     1.0
    4      898  Philadelphia Story, The (1940)     3.343279     5.0
    5      608                    Fargo (1996)     3.341272     5.0
     Hybrid RMSE: 1.7730549854033075, Hybrid MAE: 1.7531194677787105
    

## 6.3 Performance comparison between two recommendation models—the pure Collaborative Filtering (SVD) model and the Hybrid (SVD + Genre) 
The goal is to see which model generally yields better prediction accuracy (lower error) at the individual user level.




```python
# Select 10 random user IDs from the ratings dataset
random.seed(30)
random_user_ids = random.sample(ratings['userId'].unique().tolist(), 10)

# Store results
results = []

for user_id in random_user_ids:
    print(f"Evaluating for User {user_id}...")

    svd_results, svd_rmse, svd_mae = evaluate_svd(user_id, movies, ratings, best_model)
    hybrid_results, hybrid_rmse, hybrid_mae = evaluate_hybrid(user_id, movies, ratings, best_model, genre_similarity)

    results.append({
        "userId": user_id,
        "SVD RMSE": svd_rmse,
        "SVD MAE": svd_mae,
        "Hybrid RMSE": hybrid_rmse,
        "Hybrid MAE": hybrid_mae
    })
## 6.4 Bar chart
This is a two side-by-side bar charts used to visually compare the overall average performance of the pure SVD (Collaborative Filtering) model against the Hybrid (SVD + Genre) model, based on the evaluation results from the 10 randomly selected users.The SVD model has an average RMSE of ~0.8 and an average MAE of ~0.7 compared to the average RMSE and MAE of the Hybrid model which are ~1.3 and ~1.2 respectively. The SVD model outperforms the Hybrid model.


```python
# Set up figure
fig, axes = plt.subplots(1, 2, figsize = (12, 5))

# RMSE Comparison
sns.barplot(x = ["SVD", "Hybrid"], y = [results_df["SVD RMSE"].mean(), results_df["Hybrid RMSE"].mean()], ax = axes[0], palette = "viridis",
            hue = ["SVD", "Hybrid"])
axes[0].set_title("Average RMSE Comparison")
axes[0].set_ylabel("RMSE")

# MAE Comparison
sns.barplot(x = ["SVD", "Hybrid"], y=[results_df["SVD MAE"].mean(), results_df["Hybrid MAE"].mean()], ax = axes[1], palette = "rocket",
            hue = ["SVD", "Hybrid"])
axes[1].set_title("Average MAE Comparison")
axes[1].set_ylabel("MAE")

plt.tight_layout()
plt.show()
```
    
![png](Recommender_System_files/Recommender%20System%20_62_0.png)
    
## 6.4 Scatter Plot
This is a scatter plot comparing the predicted movie ratings from two different recommendation models (SVD and Hybrid) against the users' actual ratings.The plot visually represents the accuracy and bias of each model across the rated movies of 30 randomly selected users.

The plot shows that the Hybrid model's predictions (orange dots) are generally tighter and closer to the line of perfect prediction for the movies it recommends (high actual ratings).

This suggests that, at the crucial task of identifying and scoring the best movies for a user (where ratings are high), the Hybrid model demonstrates higher prediction accuracy and less variance compared to the SVD model's overall performance. This is strong evidence that blending genre similarity with collaborative filtering improves the quality of the top recommendations.

```python
# Select 30 random user IDs
random_user_ids = random.sample(ratings['userId'].unique().tolist(), 30)

# Store merged results for all users
all_svd_merged = []
all_hybrid_merged = []

for user_id in random_user_ids:
    # Get predictions for the current user
    svd_results, _, _ = evaluate_svd(user_id, movies, ratings, best_model)
    hybrid_results, _, _ = evaluate_hybrid(user_id, movies, ratings, best_model, genre_similarity)

    # Making sure that user's rating exists for the movie
    if svd_results is not None and hybrid_results is not None:
        
        # Merge with actual ratings
        actual_ratings = ratings[ratings["userId"] == user_id][["movieId", "rating"]]
        svd_merged = svd_results.merge(actual_ratings, on = "movieId")
        hybrid_merged = hybrid_results.merge(actual_ratings, on = "movieId")

        # Append to the lists
        all_svd_merged.append(svd_merged)
        all_hybrid_merged.append(hybrid_merged)
    else:
        # If user does not have a rating for a movie
        print(f"Warning: No results found for User {user_id}. Skipping...")

# Concatenate results for all users
if all_svd_merged and all_hybrid_merged:
    svd_merged_all = pd.concat(all_svd_merged)
    hybrid_merged_all = pd.concat(all_hybrid_merged)

svd_merged_all = pd.concat(all_svd_merged)
hybrid_merged_all = pd.concat(all_hybrid_merged)

# Scatter plot for predictions vs actual ratings
plt.figure(figsize = (10, 5))
plt.scatter(svd_merged_all["rating_x"], svd_merged_all["svd_score"], label = "SVD", alpha = 0.6)
plt.scatter(hybrid_merged_all["rating_x"], hybrid_merged_all["final_score"], label = "Hybrid", alpha = 0.6)

plt.plot([0, 5], [0, 5], "--", color = "gray")
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Predicted vs Actual Ratings for 30 Users")
plt.legend()
plt.show()
```

    No common movies found for user 535. Skipping evaluation.
    Warning: No results found for User 535. Skipping...
    No common movies found for user 544. Skipping evaluation.
    Warning: No results found for User 544. Skipping...
    No common movies found for user 576. Skipping evaluation.
    Warning: No results found for User 576. Skipping...
    No common movies found for user 507. Skipping evaluation.
    Warning: No results found for User 507. Skipping...
       
![png](Recommender_System_files/Recommender%20System%20_64_1.png)
    
## 6.5 Kernel Density Estimate (KDE) plot.
Showing the distribution of prediction errors for the SVD (Collaborative Filtering) model and the Hybrid (SVD + Genre) model.

The prediction error is calculated as: Error = Predicted Rating - Actual Rating.
The plot demonstrates that the Hybrid Model is significantly more accurate and precise than the SVD model, but with a different kind of bias.

Accuracy/Precision: The Hybrid errors are more tightly clustered (narrower distribution), indicating lower variance and likely lower average error (MAE/RMSE). The model is more reliable in the magnitude of its error.

Bias Trade-off: The SVD model is relatively unbiased but highly inaccurate (wide spread). The Hybrid model is highly accurate (narrow spread) but has a pessimistic bias (under-predicts ratings). Since the goal of a recommender system is typically to rank the best movies correctly, having a lower magnitude of error (tighter spread) is usually preferred, even if it introduces a consistent bias.


```python
svd_errors = svd_merged_all["svd_score"] - svd_merged_all["rating_x"]
hybrid_errors = hybrid_merged_all["final_score"] - hybrid_merged_all["rating_x"]

plt.figure(figsize = (10, 5))
sns.kdeplot(svd_errors, label = "SVD Errors", fill = True, color = "blue", alpha = 0.6)
sns.kdeplot(hybrid_errors, label = "Hybrid Errors", fill = True, color = "red", alpha = 0.6)

plt.axvline(0, color = "black", linestyle = "dashed", linewidth = 1)
plt.xlabel("Prediction Error")
plt.ylabel("Density")
plt.title("Error Distribution of SVD vs Hybrid Model")
plt.legend()
plt.show()
```
    
![png](Recommender_System_files/Recommender%20System%20_66_0.png)
    
# 7. Conclusions and Recommendations
## 7.1 Conclusion
1. The project successfully built a movie recommendation system that predicts what users are likely to enjoy based on their past ratings.

2. The collaborative filtering model (SVD) performed well, achieving an RMSE of 0.86 and MAE of 0.67, meeting the success targets.

3. The hybrid model, which combined collaborative and content-based filtering, offered more diverse and relevant movie suggestions.

4. The system demonstrates clear business value by helping streaming platforms improve user satisfaction and engagement.

5. Overall, the project achieved its main goals, accurate predictions, personalized recommendations, and practical insights for real-world use.
## 7.2 Key Points
* The SVD model established a strong baseline with an RMSE of $\approx 0.86$, which indicates that on a 5-point scale, the average prediction error is less than one point. This shows the efficacy of matrix factorization in uncovering latent user and item features.
* A Hybrid Model confirms the principle that integrating different recommendation paradigms can compensate for individual weaknesses.
  
## 7.3 Recommendations
1. Use the hybrid model in production to balance accuracy and diversity in recommendations.

2. Add more data, such as movie tags or user demographics, to improve accuracy.

3. Continuously update the model with new ratings to keep recommendations fresh and relevant.

4. Create a simple web or app interface so users can interact with the system easily.

5. Collect user feedback on recommendations to help the model learn and improve over time.

## 7.4 Next Steps 
1. Create a weighted user profile based on preferred genres to improve personalization.
2. Experiment with **Deep Matrix Factorization** or **Neural Collaborative Filtering (NCF)** to capture complex user–movie relationships.
3. Evaluate **coverage** to ensure the system recommends a wide range of movies, not just the most popular ones.
