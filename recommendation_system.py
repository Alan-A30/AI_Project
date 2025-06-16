# Sistema de Recomendaciones con IA
# Importar pandas, numpy, sklearn para crear un sistema de recomendaciones

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class RecommendationSystem:
    def __init__(self):
        self.user_item_matrix = None
        self.model = None
        self.users = None
        self.items = None

    def prepare_data(self, df):
        """
        Prepara los datos creando una matriz usuario-item.
        """
        self.users = df['user'].unique()
        self.items = df['item'].unique()
        self.user_item_matrix = df.pivot_table(index='user', columns='item', values='rating').fillna(0)
        return self.user_item_matrix

    def train_model(self):
        """
        Entrena el modelo usando similitud de coseno.
        """
        if self.user_item_matrix is None:
            raise ValueError("Primero prepara los datos.")
        self.model = cosine_similarity(self.user_item_matrix)
        return self.model

    def get_recommendations(self, user, top_n=3):
        """
        Obtiene recomendaciones para un usuario dado.
        """
        if self.model is None:
            raise ValueError("Primero entrena el modelo.")
        if user not in self.user_item_matrix.index:
            raise ValueError("Usuario no encontrado.")
        user_idx = self.user_item_matrix.index.get_loc(user)
        sim_scores = list(enumerate(self.model[user_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Ignorar el propio usuario
        sim_scores = [score for score in sim_scores if score[0] != user_idx]
        top_users = [self.user_item_matrix.index[i] for i, _ in sim_scores[:top_n]]
        # Promediar las valoraciones de los usuarios similares
        recommendations = self.user_item_matrix.loc[top_users].mean().sort_values(ascending=False)
        # Filtrar items ya vistos
        seen_items = self.user_item_matrix.loc[user][self.user_item_matrix.loc[user] > 0].index
        recommendations = recommendations.drop(seen_items, errors='ignore')
        return recommendations.head(top_n)

def generate_sample_data():
    """
    Genera un DataFrame de ejemplo con usuarios, items y ratings.
    """
    np.random.seed(42)
    users = [f'User{i}' for i in range(1, 6)]
    items = [f'Item{j}' for j in range(1, 6)]
    data = []
    for user in users:
        for item in items:
            rating = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.2, 0.1, 0.2, 0.2, 0.2, 0.1])
            data.append({'user': user, 'item': item, 'rating': rating})
    df = pd.DataFrame(data)
    return df

def main():
    # Generar datos de ejemplo
    df = generate_sample_data()
    print("Datos de ejemplo:")
    print(df.head())

    # Visualización: matriz de ratings
    pivot = df.pivot_table(index='user', columns='item', values='rating')
    plt.figure(figsize=(6,4))
    plt.title("Matriz de Ratings Usuario-Item")
    plt.imshow(pivot, cmap='viridis', aspect='auto')
    plt.colorbar(label='Rating')
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel('Items')
    plt.ylabel('Usuarios')
    plt.show()

    # Crear y usar el sistema de recomendaciones
    rec_sys = RecommendationSystem()
    rec_sys.prepare_data(df)
    rec_sys.train_model()
    user = 'User1'
    recommendations = rec_sys.get_recommendations(user)
    print(f"Recomendaciones para {user}:")
    print(recommendations)

    # Visualización: recomendaciones
    recommendations.plot(kind='bar', title=f"Recomendaciones para {user}")
    plt.ylabel('Rating estimado')
    plt.show()

if __name__ == "__main__":
    main()