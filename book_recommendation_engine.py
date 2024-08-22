# Название файла: book_recommendation_engine.py

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# Загрузка и предварительная обработка данных
def load_and_preprocess_data():
    # Загрузка данных
    ratings = pd.read_csv('/path/to/Book-Crossings/Book-Ratings.csv', delimiter=';', encoding='latin-1')
    books = pd.read_csv('/path/to/Book-Crossings/Books.csv', delimiter=';', encoding='latin-1')

    # Фильтрация пользователей и книг с достаточным количеством рейтингов
    user_counts = ratings['User-ID'].value_counts()
    book_counts = ratings['ISBN'].value_counts()
    valid_users = user_counts[user_counts >= 200].index
    valid_books = book_counts[book_counts >= 100].index
    ratings = ratings[ratings['User-ID'].isin(valid_users) & ratings['ISBN'].isin(valid_books)]

    # Объединение данных
    book_info = books[['ISBN', 'Book-Title']]
    data = pd.merge(ratings, book_info, on='ISBN')

    return data


# Создание модели рекомендаций
def create_model(data):
    # Векторизация названий книг
    vectorizer = TfidfVectorizer()
    book_titles = data['Book-Title'].unique()
    book_title_vectors = vectorizer.fit_transform(book_titles)

    # Построение модели KNN
    model = NearestNeighbors(n_neighbors=6, metric='cosine')
    model.fit(book_title_vectors)

    return model, vectorizer, book_titles


# Функция для получения рекомендаций
def get_recommends(title, model, vectorizer, book_titles):
    # Преобразование названия книги в вектор
    title_vector = vectorizer.transform([title])

    # Поиск ближайших соседей
    distances, indices = model.kneighbors(title_vector)

    # Получение результатов
    recommendations = []
    for i in range(1, len(distances[0])):
        recommended_title = book_titles[indices[0][i]]
        recommendations.append([recommended_title, distances[0][i]])

    return [title, recommendations]


# Загрузка и обработка данных
data = load_and_preprocess_data()
model, vectorizer, book_titles = create_model(data)


# Тестирование функции
def test_function():
    test_title = "The Queen of the Damned (Vampire Chronicles (Paperback))"
    recommendations = get_recommends(test_title, model, vectorizer, book_titles)
    print(recommendations)


test_function()
