import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data = pd.read_csv("movies.csv")

# Select the relevant features
selected_features = ["genres", "keywords", "tagline", "cast", "director"]

# Replace null values with empty strings
for feature in selected_features:
    data[feature] = data[feature].fillna("")

# Combine the features into one column
combined_features = (
    data["genres"]
    + " "
    + data["keywords"]
    + " "
    + data["tagline"]
    + " "
    + data["cast"]
    + " "
    + data["director"]
)

# Convert text data into feature vectors using TfidfVectorizer
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Compute the similarity matrix using cosine similarity
similarity = cosine_similarity(feature_vectors)

# List of all movie titles for reference
list_of_all_titles = data["title"].tolist()

# Streamlit app UI with enhanced styling
st.title("🎬 Movie Recommendation System")
st.markdown("### Find similar movies based on your favorite film! 🎥🍿")

# Main app layout with form to capture Enter key submission
with st.form("recommendation_form"):
    # Input for the user's favorite movie
    movie_name = st.text_input("Enter your favorite movie title:")

    # Slider to select the number of recommendations
    num_recommendations = st.slider(
        "Select number of recommendations:", min_value=5, max_value=20, value=10
    )

    # Submit button (works on Enter key or button click)
    submit_button = st.form_submit_button("Recommend")

if submit_button:
    if movie_name:
        # Find the closest match for the input movie
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

        if find_close_match:
            close_match = find_close_match[0]

            # Displaying the main movie name with enhanced visibility
            st.markdown(f"### Recommendations based on:")
            st.markdown(
                f"<h2 style='color: #FF6347; font-size: 24px;'>{close_match}</h2>",
                unsafe_allow_html=True,
            )

            # Get the index of the matched movie
            index_of_movie = data[data.title == close_match].index[0]

            # Get a list of similar movies based on the similarity score
            similarity_score = list(enumerate(similarity[index_of_movie]))
            sorted_similar_movies = sorted(
                similarity_score, key=lambda x: x[1], reverse=True
            )

            # Display recommendations in a nicely formatted list
            st.markdown("#### 🎉 Recommended Movies for You:")
            st.write(f"Here are your top {num_recommendations} recommendations:")

            # Display recommendations in a grid format for visual appeal
            num_cols = 2
            recommended_movies = [
                data.iloc[movie[0]].title
                for movie in sorted_similar_movies[1 : num_recommendations + 1]
            ]
            for i in range(0, len(recommended_movies), num_cols):
                cols = st.columns(num_cols)
                for idx, col in enumerate(cols):
                    if i + idx < len(recommended_movies):
                        col.markdown(f"**🎬 {recommended_movies[i + idx]}**")
        else:
            st.error("Sorry, no match found. Please check the movie name.")
    else:
        st.warning("Please enter a movie name to get recommendations.")
