import joblib
import pandas as pd
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load the model and vectorizer
cosine_similarities = joblib.load('cosine_similarities_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer_model.joblib')
tfidf_matrix_train = joblib.load('tfidf_matrix_train.joblib')
# Load anime data
# Assuming df is your dataframe with anime information
df = pd.read_csv('AnimeList.csv')  # Adjust the file name and path accordingly


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_genre = request.form['user_genre']
        user_episodes = request.form['user_episodes']
        user_score = request.form['user_score']
        user_input = f"{user_genre}{user_episodes}{user_score}"
        # Transform user input using the TF-IDF vectorizer
        user_input_tfidf = tfidf_vectorizer.transform([user_input])

        # Calculate cosine similarity between user input and all anime
        cosine_similarities_user = linear_kernel(user_input_tfidf, tfidf_matrix_train).flatten()

        # Get indices of anime with high similarity
        similar_anime_indices = cosine_similarities_user.argsort()[:-6:-1]
        print("User Input:", user_input)
        print("Similar Anime Indices:", similar_anime_indices)
        print("Cosine Similarities:", cosine_similarities_user)

        # Get recommended anime names
        recommended_anime_info = df.iloc[similar_anime_indices][['title','genre','title_english']].to_dict(orient='records')
        print("Recommended Anime Info:", recommended_anime_info)
        return render_template('recommendation.html', recommendation=recommended_anime_info)

    except KeyError as e:
        # Handle the case where one of the form fields is missing
        error_message = f"Error: {e} is missing in the form data."
        return render_template('error.html', error_message=error_message)
    except Exception as e:
        # Handle other exceptions
        error_message = f"An error occurred: {str(e)}"
        return render_template('error.html', error_message=error_message)


if __name__ == '__main__':
    app.run(debug=True)
