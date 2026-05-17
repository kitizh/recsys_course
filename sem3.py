"""
Семинар 3. Контентная фильтрация
Цель: Разработать методы контентной фильтрации по пользователям и по фильмам.
В качестве контента используем описание жанров для каждого фильма из movies.csv.
Для векторизации жанров используем CountVectorizer с разделителем "|".
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from utils import build_user_item_matrix, id_to_movie, load_data, print_user_rated_items


class ContentRecommender:
    """
    Класс для построения рекомендаций на основе контента - описания жанров.
    Матрица эмбеддингов размером (max_movie_id+1, n_genres), где строки
    соответствуют movieId, а столбцы — one-hot кодированию жанров.
    Матрица строится при инициализации экземпляра класса.
    """

    def __init__(self):
        self.embeddings = None
        self.ui_matrix = build_user_item_matrix()
        self._build_embeddings()

    def _build_embeddings(self):
        _, movies_df = load_data()
        self.movies_df = movies_df.copy()
        self.movies_df["genres"] = self.movies_df["genres"].fillna("")
        vectorizer = CountVectorizer(
            tokenizer=lambda s: s.split("|"), lowercase=False, token_pattern=None
        )
        ###########################################################################
        # векторизуем строку жанров для каждого фильма
        genre_vectors = vectorizer.fit_transform(self.movies_df["genres"]).toarray()

        # создаем матрицу, где индекс строки совпадает с movieId
        max_movie_id = max(self.movies_df["movieId"].max() + 1, self.ui_matrix.shape[1])
        self.embeddings = np.zeros((max_movie_id, genre_vectors.shape[1]))

        # переносим жанровые векторы в строки с нужными movieId
        for idx, movie_id in enumerate(self.movies_df["movieId"]):
            self.embeddings[int(movie_id)] = genre_vectors[idx]
        ###########################################################################

    def predict_rating(self, user_id: int, item_id: int, k: int = 5) -> float:
        """
        Предсказывает рейтинг user_id для item_id на основе контентной фильтрации.

        Алгоритм:
        1) Берём вектор целевого фильма: target_vec.
        2) Находим все фильмы, оцененные пользователем.
        3) Считаем косинусное сходство target_vec с векторами оцененных фильмов.
        4) Отбираем топ-k похожих оцененных фильмов (k-параметр).
        5) Предсказываем рейтинг как взвешенное среднее оценок по сходствам.
        6) Если не удаётся предсказать (нет оценок или нулевые векторы), возвращаем 0.0.
        7) Клипируем результат в [0.0, 5.0].

        Args:
            user_id: индекс пользователя
            item_id: индекс фильма
            k: сколько наиболее похожих оцененных фильмов использовать

        Returns:
            float: предсказанный рейтинг
        """
        user_ratings = self.ui_matrix[user_id]

        # берем вектор фильма, для которого нужен прогноз
        target_vec = self.embeddings[item_id]
        target_norm = np.linalg.norm(target_vec)
        if target_norm == 0:
            return 0.0

        # находим фильмы, которые пользователь уже оценил
        rated_items = np.where(user_ratings > 0)[0]
        if len(rated_items) == 0:
            return 0.0

        # считаем сходство целевого фильма с оцененными фильмами
        rated_vectors = self.embeddings[rated_items]
        dot_products = rated_vectors @ target_vec
        norms = np.linalg.norm(rated_vectors, axis=1) * target_norm
        similarities = np.divide(
            dot_products,
            norms,
            out=np.zeros_like(dot_products, dtype=float),
            where=norms != 0,
        )

        # берем k самых похожих оцененных фильмов
        sorted_idx = np.argsort(similarities)[::-1][:k]
        top_similarities = similarities[sorted_idx]
        top_ratings = user_ratings[rated_items][sorted_idx]

        # оставляем только ненулевое сходство
        positive_mask = top_similarities > 0
        if not np.any(positive_mask):
            return 0.0
        top_similarities = top_similarities[positive_mask]
        top_ratings = top_ratings[positive_mask]

        # считаем прогноз как взвешенную среднюю оценку
        prediction = np.dot(top_similarities, top_ratings) / top_similarities.sum()
        return float(np.clip(prediction, 0.0, 5.0))

    def predict_items_for_user(
        self, user_id: int, k: int = 5, n_recommendations: int = 5
    ) -> list:
        """
        Рекомендует фильмы пользователю user_id на основе контента фильма.

        Алгоритм:
        1) Берем все фильмы, которые оценил пользователь.
        3) Строим профиль пользователя как взвешенное среднее жанров оцененных фильмов.
        4) Для всех фильмов, которые пользователь не оценил, считаем сходство с профилем.
        5) Сортируем по убыванию сходства и возвращаем top-n.
        """
        user_ratings = self.ui_matrix[user_id]

        # берем фильмы, которые уже есть в истории пользователя
        rated_items = np.where(user_ratings > 0)[0]
        if len(rated_items) == 0:
            return []

        # строим профиль пользователя по жанрам с весами из оценок
        rated_vectors = self.embeddings[rated_items]
        ratings = user_ratings[rated_items]
        if ratings.sum() == 0:
            return []
        user_profile = (rated_vectors * ratings[:, None]).sum(axis=0) / ratings.sum()

        # если профиль пустой, рекомендовать нечего
        profile_norm = np.linalg.norm(user_profile)
        if profile_norm == 0:
            return []

        # выбираем фильмы, которые пользователь еще не оценивал
        unrated_items = np.where(user_ratings == 0)[0]
        if len(unrated_items) == 0:
            return []

        # считаем сходство каждого кандидата с профилем пользователя
        unrated_vectors = self.embeddings[unrated_items]
        dot_products = unrated_vectors @ user_profile
        norms = np.linalg.norm(unrated_vectors, axis=1) * profile_norm
        similarities = np.divide(
            dot_products,
            norms,
            out=np.zeros_like(dot_products, dtype=float),
            where=norms != 0,
        )

        # сортируем кандидатов и берем top-n
        sorted_idx = np.argsort(similarities)[::-1][:n_recommendations]
        recommendations = unrated_items[sorted_idx]

        return [int(item_id) for item_id in recommendations]


# Пример использования для дебага:
if __name__ == "__main__":
    user_id = 10
    item_id = 2
    k = 5
    content_recommender = ContentRecommender()
    print_user_rated_items(user_id, content_recommender.ui_matrix)

    pred_rating = content_recommender.predict_rating(user_id, item_id, k)
    print(f"Predicted rating for user {user_id} and item {item_id}: {pred_rating:.2f}")

    recommendations = content_recommender.predict_items_for_user(
        user_id, k=5, n_recommendations=10
    )
    for rec in recommendations:
        print(f"Recommended movie ID: {rec}, Title: {id_to_movie(rec)}")
