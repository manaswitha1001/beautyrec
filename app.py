import streamlit as st
import pandas as pd
import numpy as np 
import base64
from tabulate import tabulate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from surprise import Dataset, Reader, KNNBasic, SVD, NMF, dump
from surprise.model_selection import train_test_split


@st.cache_data
def load_data():
    df_product_info = pd.read_csv("https://raw.githubusercontent.com/manaswitha1001/beautyrecommendationapp/main/datasets/product_info.csv", header=0)
    df_product = df_product_info[['product_id', 'product_name', 'brand_name', 'ingredients', 'highlights', 'primary_category', 'secondary_category', 'tertiary_category']]
    df_product['tertiary_category'].fillna(df_product['secondary_category'], inplace=True)
    df_product = df_product.dropna()

    df_reviews = pd.read_csv('https://raw.githubusercontent.com/manaswitha1001/beautyrecommendationapp/main/datasets/reviews.csv', header=0, low_memory=False)
    df_review = df_reviews[['author_id', 'product_id', 'rating']]
    df_review = df_review.drop_duplicates()

    return df_product_info, df_product, df_review

print("loadign data")

df_product_info, df_product, df_review = load_data()

reader = Reader(rating_scale=(0, 5))
review_data = Dataset.load_from_df(df_review, reader)
trainset, _ = train_test_split(review_data, test_size=.2, random_state=42)

svd_best_model = SVD(n_factors=100, n_epochs=50, lr_all=0.01, reg_all=0.1)
svd_best_model.fit(trainset)


# Utility Functions 
def get_recommendations(product_id, top_n):
    """
    Get product recommendations based on similarity.

    Parameters:
    - product_id (str): The ID of the product for which recommendations are sought.
    - top_n (int): The number of top recommendations to retrieve.

    Returns:
    pandas.DataFrame: A DataFrame containing the top recommendations with additional information.
    """
    # Find the index of the given product_id in the DataFrame
    product_index = df_product[df_product['product_id'] == product_id].index[0]

    # Retrieve the similarity scores for the given product
    similarities = similarity_matrix[product_index]

    # Find the indices of the most similar products (excluding the product itself)
    similar_indices = np.argsort(similarities)[::-1][1:top_n+1]

    # Extract relevant information for the recommended products
    recommendations = df_product.loc[similar_indices, ['product_id', 'product_name', 'brand_name', 'tertiary_category']]

    # Renmae the columns
    column_mapping = {
        'product_id': 'ID',
        'product_name': 'Product Name',
        'brand_name': 'Brand',
        'tertiary_category': 'Category'
    }
    recommendations = recommendations.rename(columns=column_mapping)
    
    # Add a 'rank' column to the recommendations
    recommendations.insert(0, 'Rank', range(1, top_n+1))

    # Return the recommendations DataFrame
    return recommendations


def get_rated_products_by_user(user_id, df_review, df_product_info):
    """
    Get the list of products that a user has interacted with and rated highly.

    Parameters:
    - user_id (int): The ID of the user for whom rated products are being retrieved.
    - df_review (pandas.DataFrame): The DataFrame containing review information.
    - df_product_info (pandas.DataFrame): The DataFrame containing product information.

    Returns:
    pandas.DataFrame: A DataFrame containing product details with renamed columns.
    """
    # Filter reviews for the given user with ratings >= 4.0
    user_ratings = df_review[(df_review['author_id'] == user_id) & (df_review['rating'] >= 4.0)]

    # Merge with product information using the 'product_id' column
    merged_data = pd.merge(user_ratings, df_product_info, on='product_id')

    # Display a message indicating products with high ratings by the user
    print("\nProducts with High Ratings by User", user_id, ":")

    # Extract relevant columns from the merged data
    product_details = merged_data[['product_id', 'product_name', 'brand_name', 'primary_category', 'rating_x']]

    # Reset the index of the resulting DataFrame
    product_details = product_details.reset_index(drop=True)

    # Rename the columns
    column_mapping = {
        'product_id': 'ID',
        'product_name': 'Product Name',
        'brand_name': 'Brand',
        'primary_category': 'Category',
        'rating_x': 'User_Rating'
    }
    product_details = product_details.rename(columns=column_mapping)

    # Return the DataFrame with renamed columns
    return product_details



def get_user_recommendations(user_id, model, df_review, df_product_info, n):
    """
    Get a list of potential product recommendations for a user based on collaborative filtering.

    Parameters:
    - user_id (int): The ID of the user for whom recommendations are being generated.
    - model: The collaborative filtering model used for predictions.
    - df_review (pandas.DataFrame): The DataFrame containing review information.
    - df_product_info (pandas.DataFrame): The DataFrame containing product information.
    - n (int, optional): The number of top recommendations to retrieve. Default is 5.

    Returns:
    pandas.DataFrame: A DataFrame containing top-N product recommendations with details.
    """
    # Get a list of all unique product IDs
    all_product_ids = df_review['product_id'].unique()

    # Remove products that the user has already rated
    products_rated_by_user = df_review[df_review['author_id'] == user_id]['product_id'].values
    products_to_predict = np.setdiff1d(all_product_ids, products_rated_by_user)

    # Make predictions for the products to predict
    predictions = [model.predict(user_id, product_id) for product_id in products_to_predict]

    # Sort the predictions by estimated ratings (in descending order)
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Get the top-N recommended products
    top_n_recommendations = [prediction.iid for prediction in predictions[:n]]

    # Populate the recommendations list
    recommendations = []
    for i, product_id in enumerate(top_n_recommendations, start=1):
        # Find the product details in df_product_info
        product_details = df_product_info[df_product_info['product_id'] == product_id]

        if not product_details.empty:
            product_name = product_details['product_name'].values[0]
            brand_name = product_details['brand_name'].values[0]
            primary_category = product_details['primary_category'].values[0]

            # Append the recommendation to the list
            recommendations.append([i, product_id, product_name, brand_name,primary_category])

    columns = ['Rank', 'Product_ID', 'Product Name', 'Brand', 'Category']
    recommendations_df = pd.DataFrame(recommendations, columns=columns)

    return recommendations_df


#TFIDF model 
@st.cache_data
def compute_similarity_matrix(df_product):
    tfidf = TfidfVectorizer()
    ingredient_vector = tfidf.fit_transform(df_product['ingredients'])
    highlights_vector = tfidf.fit_transform(df_product['highlights'])
    prim_category_vector = tfidf.fit_transform(df_product['primary_category'])
    sec_category_vector = tfidf.fit_transform(df_product['secondary_category'])
    tert_category_vector = tfidf.fit_transform(df_product['tertiary_category'])
    # Combine ingredient , highlights and category vectors
    feature_vectors = hstack((ingredient_vector,highlights_vector, prim_category_vector,sec_category_vector, tert_category_vector))
    similarity_matrix = cosine_similarity(feature_vectors)
    return similarity_matrix

df_product = df_product.reset_index(drop=True)
similarity_matrix = compute_similarity_matrix(df_product)

##SVD MODEL

best_params = {'n_factors': 100, 'n_epochs': 50, 'lr_all': 0.01, 'reg_all': 0.1}
svd_best_model = SVD(**best_params)
svd_best_model.fit(review_data.build_full_trainset())
dump.dump('svd_model', algo=svd_best_model)
_, svd_best_model = dump.load('svd_model')

print("iii")


# Add background image 
def add_bg_from_github(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
            
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Example usage:
background_image_url = 'https://raw.githubusercontent.com/manaswitha1001/beautyrecommendationapp/6053306d2930a7d471fad19b039746131e79c6e4/images/makeup.jpg'
add_bg_from_github(background_image_url)


# Side bar 
preference = st.sidebar.slider('Choose number of recommendations', 0, 10, 5)
page_options = ['FavFinder','TraitPicker','SquadSuggester', 'BlendBuddy']
choice = st.sidebar.selectbox('Select Page', page_options)


if choice == 'FavFinder':
    st.title('Sephora Beauty Recommender')
    st.title('Most loved Products')
    selected_category = st.selectbox('Category', df_product_info['tertiary_category'].unique())
    df_selected_category = df_product_info[df_product_info['tertiary_category']== selected_category]
    df_sorted = df_selected_category.sort_values(by='loves_count', ascending=False)
    top_products = df_sorted[['product_name','brand_name']].head(n= preference)
    columns_map = {'product_name':'Product','brand_name':'Brand'}
    top_products.rename(columns = columns_map, inplace = True)
    st.dataframe(top_products, hide_index=True)

elif choice == 'TraitPicker':
    st.title('Trait Picker')
    col1, col2 = st.columns(2)
    # Dropdown for selecting brand
    with col1:
        selected_brand = st.selectbox('Brand', df_product['brand_name'].unique())
    # Filter products based on the selected brand
    filtered_products = df_product[df_product['brand_name'] == selected_brand]['product_name']
    # Dropdown for selecting product
    with col2:
        selected_product = st.selectbox('Product', filtered_products)
    # Display the selected brand and product
    st.write(f'Selected Brand: {selected_brand}')
    st.write(f'Selected Product: {selected_product}')

    info_product = df_product[(df_product['product_name'] == selected_product) & (df_product['brand_name'] == selected_brand)][['product_id','product_name', 'brand_name', 'primary_category','secondary_category' ,'tertiary_category']]
    product_table = tabulate(info_product, headers='keys', tablefmt='psql',showindex=False )
    # similarity_matrix = compute_similarity_matrix()
    # Button to get recommendations
    if st.button('Get Similar Recommendations'):
        info_product = df_product[(df_product['product_name'] == selected_product) & (df_product['brand_name'] == selected_brand)][['product_id','product_name', 'brand_name', 'tertiary_category']]
        product_table = tabulate(info_product, headers='keys', tablefmt='psql',showindex=False )
        selected_product_id= info_product['product_id'].values[0]
        selected_product_info = info_product[['product_name', 'brand_name', 'tertiary_category']]
        columns_map = {'product_name':'Product','brand_name':'Brand', 'tertiary_category': 'Category'}
        selected_product_info.rename(columns = columns_map, inplace = True)
        st.dataframe(selected_product_info, hide_index=True)
        # similarity_matrix = compute_similarity_matrix()
        recommendations = get_recommendations(selected_product_id, top_n=preference)
        st.write('Similar Product Recommendations')
        st.dataframe(recommendations, hide_index=True)

elif choice == 'SquadSuggester':
    st.title('Squad Suggester')
    user_input = st.text_input('Enter the user id here:', '8447021668')
    if st.button("Get squad suggestions"):
        rated_products = get_rated_products_by_user(user_input, df_review, df_product_info)
        st.write("The products rated by user ", user_input)
        st.dataframe(rated_products, hide_index=True)
        recs = get_user_recommendations(user_input, svd_best_model, df_review, df_product_info, n=preference)
        st.write("The products recommended for user ", user_input)
        st.dataframe(recs, hide_index=True)


# elif choice == 'Blend Buddy':
#     st.title('Blend Buddy')
#     user_id = st.text_input('Enter the user id here:', '8447021668')
#     col1, col2 = st.columns(2)
#     # Dropdown for selecting brand
#     with col1:
#         selected_brand = st.selectbox('Brand', df_product['brand_name'].unique())
#     # Filter products based on the selected brand
#     filtered_products = df_product[df_product['brand_name'] == selected_brand]['product_name']
#     # Dropdown for selecting product
#     with col2:
#         selected_product = st.selectbox('Product', filtered_products)
#     # Display the selected brand and product
#     st.write(f'Selected Brand: {selected_brand}')
#     st.write(f'Selected Product: {selected_product}')

#     info_product = df_product[(df_product['product_name'] == selected_product) & (df_product['brand_name'] == selected_brand)][['product_id','product_name', 'brand_name', 'primary_category','secondary_category' ,'tertiary_category']]
#     product_table = tabulate(info_product, headers='keys', tablefmt='psql',showindex=False )
  
#     if st.button('Get Recommendations'):
#         info_product = df_product[(df_product['product_name'] == selected_product) & (df_product['brand_name'] == selected_brand)][['product_id','product_name', 'brand_name', 'tertiary_category']]
#         product_table = tabulate(info_product, headers='keys', tablefmt='psql',showindex=False )
#         selected_product_id= info_product['product_id'].values[0]
#         selected_product_info = info_product[['product_name', 'brand_name', 'tertiary_category']]
#         columns_map = {'product_name':'Product','brand_name':'Brand', 'tertiary_category': 'Category'}
#         selected_product_info.rename(columns = columns_map, inplace = True)
#         st.dataframe(selected_product_info, hide_index=True)
#         # similarity_matrix = compute_similarity_matrix()
#         recommendations = get_recommendations(selected_product_id, top_n=5)
#         rated_products = get_rated_products_by_user(user_id, df_review, df_product_info)
#         st.write("The products rated by user ", user_id)
#         st.dataframe(rated_products, hide_index=True)
#         recs = get_user_recommendations(user_id, svd_best_model, df_review, df_product_info, n=5)
#         st.write("The products recommended for user ", user_id)
#         st.dataframe(recs, hide_index=True)
      
      
