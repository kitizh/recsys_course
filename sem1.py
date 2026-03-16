"""
Seminar 1
Построение простой рекомендательной системы на основе популярности товаров

Цель: Разработать простейшие рекомендательные системы,
на основе имеющихся данных о взаимодействии с фильмами,
проанализировать их эффективность.
"""

import numpy as np

from utils import load_data


def random_recommend(n_recommendations: int = 10, seed: int = 42) -> list[int]:
    """
    Рекомендует случайные фильмы.

    Args:
        n_recommendations: Количество рекомендаций
        seed: Seed для воспроизводимости

    Returns:
        Список ID фильмов
    """
    ratings_df, _ = load_data()
    np.random.seed(seed)
    recommendations = []

    ### Ваш код здесь ###
    recommendations = ratings_df["movieId"].unique()
    recommendations = np.random.choice(
        recommendations, size=n_recommendations, replace=False
    )
    recommendations = recommendations.tolist()
    ### Конец вашего кода ###

    return recommendations


def top_n_recommend(
    n_recommendations: int = 10, min_ratings: int = 10
) -> list[tuple[int, float, int]]:
    """
    Рекомендует самые популярные фильмы на основе среднего рейтинга и количества оценок.

    Args:
        n_recommendations: Количество рекомендаций
        min_ratings: Минимальное количество рейтингов для фильма

    Returns:
        Список кортежей (movieId, avg_rating, rating_count)
    """
    ratings_df, movies_df = load_data()
    top_n_recs = []

    ### Ваш код здесь ###
    movie_stats = (
        ratings_df.groupby("movieId")["rating"]
        .agg(["mean", "count"])  # средняя оценка и общее число оценок
        .reset_index()
    )  # получится таблица вида movieId | mean | count

    # оставляем фильмы с количеством рейтингов выше минимального
    movie_stats = movie_stats[movie_stats["count"] >= min_ratings]

    # сортируем по убыванию
    movie_stats = movie_stats.sort_values(
        by=["mean", "count"], ascending=[False, False]
    )

    # брем верхние n_recommendations записей и каждую запись записываем как кортеж
    top_n_recs = list(
        movie_stats.head(n_recommendations)[["movieId", "mean", "count"]].itertuples(
            index=False, name=None
        )
    )

    ### Конец вашего кода ###

    return top_n_recs


def evaluate_rec_systems(
    user_id: int = 610, n_recommendations: int = 10, random_state: int = 42
) -> dict:
    """
    Оценивает эффективность рекомендательной системы.
    Метрика Accuracy для двух подходов: случайные и популярные фильмы.

    Args:
        user_id: ID пользователя для оценки
        n_recommendations: Количество рекомендаций
        random_state: Seed для воспроизводимости

    Returns:
        Словарь с метрикой Accuracy для двух подходов: случайные и популярные фильмы.
    """
    ratings_df, _ = load_data()
    random_accuracy = 0.0
    popular_accuracy = 0.0
    ### Ваш код здесь ###

    # список фильмов которые пользователь оценивал
    user_movies = set(ratings_df[ratings_df["userId"] == user_id]["movieId"])

    # список id случайных n рекомендаций
    random_recs_ids = random_recommend(
        n_recommendations=n_recommendations, seed=random_state
    )

    # список n популярных рекомендаций
    popular_recs_with_stats = top_n_recommend(n_recommendations=n_recommendations)

    # оставляем только id популярных рекомендаций
    popular_recs_ids = [m_id for m_id, _, _ in popular_recs_with_stats]

    # берем пересечение
    random_hits = len(set(random_recs_ids) & user_movies)
    popular_hits = len(set(popular_recs_ids) & user_movies)

    # делим число совпадений на общее число рекомендаций
    random_accuracy = random_hits / n_recommendations
    popular_accuracy = popular_hits / n_recommendations

    ### Конец вашего кода ###
    return {"random_accuracy": random_accuracy, "popular_accuracy": popular_accuracy}


if __name__ == "__main__":
    # 1. Случайные рекомендации
    print("\n1. СЛУЧАЙНЫЕ РЕКОМЕНДАЦИИ:")
    print("-" * 60)
    random_recs = random_recommend(n_recommendations=10)
    print(f"Рекомендованные ID фильмов: {random_recs}")

    # 2. Популярные фильмы
    print("\n2. ПОПУЛЯРНЫЕ ФИЛЬМЫ (рекомендации на основе популярности):")
    print("-" * 60)
    popular_recs = top_n_recommend(n_recommendations=10)
    print(f"{'Rank':<5} {'ID':<6} {'Ср рейтинг':<18} {'Кол-во оценок':<15}")
    print("-" * 60)
    for i, (movie_id, avg_rating, rating_count) in enumerate(popular_recs, 1):
        print(f"{i:<5} {movie_id:<6} {avg_rating:<18.2f} {rating_count:<15}")

    # 3. Оценка системы
    print("\n3. ОЦЕНКА КАЧЕСТВА СИСТЕМЫ:")
    print("-" * 60)
    metrics = evaluate_rec_systems()
    print(f"Accuracy (случайные рекомендации): {metrics['random_accuracy']:.4f}")
    print(f"Accuracy (популярные фильмы): {metrics['popular_accuracy']:.4f}")
