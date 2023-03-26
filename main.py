from util import levenshtein_similarity
import numpy as np
from util import connect_db
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import matplotlib.pyplot as plt


def get_sentences_from_db(host, user, password, database):
    """
    Fetch sentences from the 'dokumen' table in the database.
    """
    conn, cursor = connect_db(host, user, password, database)
    cursor.execute("SELECT preproccess_text FROM dokumen")
    result = cursor.fetchall()
    sentences = [row[0] for row in result]
    cursor.close()
    conn.close()
    return sentences


def compute_similarity_matrix(sentences):
    """
    Compute the Levenshtein similarity matrix between all pairs of sentences.
    """
    n = len(sentences)
    similarity_matrix = [[0] * (n + 1) for i in range(n + 1)]

    # Add row and column headers
    similarity_matrix[0][1:] = range(1, n + 1)
    for i in range(n):
        similarity_matrix[i + 1][0] = i + 1

    for i in range(n):
        for j in range(i + 1, n):
            similarity = levenshtein_similarity(sentences[i], sentences[j])
            similarity = 1 - similarity
            similarity_matrix[i + 1][j + 1] = similarity
            similarity_matrix[j + 1][i + 1] = similarity

    # Return inner matrix without row and column headers
    return [row[1:] for row in similarity_matrix[1:]]


def plot_similarity_matrix(similarity_matrix):
    # Print the similarity matrix as a table
    for row in similarity_matrix:
        print("\t".join([f"{value:.2f}" if isinstance(
            value, float) else str(value) for value in row]))


def optimal_eps_min_samples(similarity_matrix):
    nbrs = NearestNeighbors(n_neighbors=5).fit(similarity_matrix)
    neigh_dist, neigh_ind = nbrs.kneighbors(similarity_matrix)
    sort_neigh_dist = np.sort(neigh_dist, axis=0)
    k_dist = sort_neigh_dist[:, 4]
    kneedle = KneeLocator(x=range(1, len(neigh_dist)+1), y=k_dist, S=1.0,
                          curve="concave", direction="increasing", online=True)

    print(kneedle.knee_y)
    print(kneedle.knee)
    return kneedle.knee_y, kneedle.knee


def find_clusters(similarity_matrix, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    dbscan.fit(similarity_matrix)
    labels = dbscan.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    for i in range(n_clusters_):
        print("Cluster", i, "contains", list(labels).count(i), "points")

    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)

    return labels


def plot_clusters(similarity_matrix, labels):
    plt.scatter(similarity_matrix[:, 0],
                similarity_matrix[:, 1], c=labels, cmap='viridis')
    plt.show()


def main():

    # sentences = get_sentences_from_db("localhost", "root", "", "deteksi_trending_topik")
    sentences = [
        "lonjakan kasus corona indonesia",
        "vaksinasi indonesia astrazeneca",
        "swab antigen vaksin sinovac",
        "vaksin corona serentak nasional",
        "isolasi mandiri masyarakat terpapar covid"
    ]

    similarity_matrix = compute_similarity_matrix(sentences)
    similarity_matrix = np.array(similarity_matrix)
    # print(similarity_matrix)
    plot_similarity_matrix(similarity_matrix)
    eps, min_samples = optimal_eps_min_samples(similarity_matrix)

    clusters = find_clusters(
        similarity_matrix, eps, min_samples)
    # clusters = cluster_kmeans(similarity_matrix)
    plot_clusters(similarity_matrix, clusters)

    print("Clusters:", clusters)


if __name__ == "__main__":
    main()
