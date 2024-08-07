import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from sklearn.metrics import silhouette_score
import time
import sys

class ShortTermMemory:
    def __init__(self, max_iter=15, random_state=92, decay_rate=0.95, boost_rate=1.1, attention_threshold=0.5):
        self.max_iter = max_iter
        self.random_state = random_state
        self.decay_rate = decay_rate
        self.boost_rate = boost_rate
        self.attention_threshold = attention_threshold
        self.centroids = None
        self.hash_table = defaultdict(list)
        self.time_complexity = []
        self.space_complexity = []

    def fit(self, X):
        # Use Bayesian Optimization to find the optimal number of clusters
        optimal_params = self.fit_cluster(X)
        n_clusters = optimal_params['n_clusters']
        
        start_time = time.time()
        
        np.random.seed(self.random_state)
        self.centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]
        self.centroids = self.centroids / np.linalg.norm(self.centroids, axis=1, keepdims=True)
        
        for iteration in range(self.max_iter):
            clusters = self._assign_clusters(X)
            new_centroids = self._update_centroids(X, clusters)
            
            if np.allclose(self.centroids, new_centroids, atol=1e-5):
                print(f"Converged after {iteration + 1} iterations")
                break
            
            self.centroids = new_centroids
            print(f"Completed iteration {iteration + 1}")
        
        self._populate_hash_table(X, clusters)
        
        end_time = time.time()
        self.time_complexity.append(end_time - start_time)
        self.space_complexity.append(self._calculate_space_complexity())

    def fit_cluster(self, X):
    # Define the search space for Bayesian optimization
        space  = [
            Integer(2, min(len(X), 8000), name='n_clusters'),  # Number of clusters, minimum 2
        ]
    
        @use_named_args(space)
        def objective(**params):
            n_clusters = params['n_clusters']
            
        
            # Initialize the centroids
            centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]
            centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        
            # Fit the model
            start_time = time.time()
            for iteration in range(self.max_iter):
                clusters = self._assign_clusters(X, centroids)
                new_centroids = self._update_centroids(X, clusters, centroids)
            
                if np.allclose(centroids, new_centroids, atol=1e-5):
                    break
            
                centroids = new_centroids
        
            end_time = time.time()
        
            # Calculate inertia
            inertia = np.sum(np.min(cosine_similarity(X, centroids), axis=1))

            # Calculate silhouette score if possible
            try:
                silhouette_avg = silhouette_score(X, clusters)
            except ValueError:
                # If silhouette score can't be calculated, use a default value
                silhouette_avg = 0
        
            # Calculate space complexity
            space_complexity = n_clusters * X.shape[1]
        
            # Calculate time complexity
            time_complexity = end_time - start_time
        
            # Define loss function (example: weighted sum of inertia and inverse silhouette score)
            loss = inertia - silhouette_avg + 0.01 * space_complexity + 0.01 * time_complexity
        
            # Add small random noise to encourage exploration
            loss += np.random.normal(0, 0.1)
        
            return loss
    
        # Perform Bayesian optimization with adjusted parameters
        res = gp_minimize(objective, space, n_calls=30, n_random_starts=20, random_state=self.random_state)
    
        print("Optimal parameters:", res.x)
        print("Minimum loss:", res.fun)
    
        return {'n_clusters': res.x[0], 'smoothness': res.x[1]}

    def _assign_clusters(self, X, centroids=None):
        if centroids is None:
            centroids = self.centroids
        normalized_X = X / np.linalg.norm(X, axis=1, keepdims=True)
        similarities = cosine_similarity(normalized_X, centroids)
        return np.argmax(similarities, axis=1)
    
    def _update_centroids(self, X, clusters, centroids=None):
        if centroids is None:
            centroids = self.centroids
        new_centroids = []
        for i in range(len(centroids)):
            cluster_points = X[clusters == i]
            if len(cluster_points) > 0:
                new_centroid = np.mean(cluster_points, axis=0)
                new_centroid /= np.linalg.norm(new_centroid)
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(centroids[i])
        return np.array(new_centroids)
    
    def _populate_hash_table(self, X, clusters):
        self.hash_table.clear()
        normalized_X = X / np.linalg.norm(X, axis=1, keepdims=True)
        for i, point in enumerate(normalized_X):
            cluster = clusters[i]
            self.hash_table[tuple(self.centroids[cluster])].append((point, 1.0))  # Initial attention of 1.0
    
    def add_vector(self, vector):
        start_time = time.time()
        
        normalized_vector = vector / np.linalg.norm(vector)
        similarities = cosine_similarity(normalized_vector.reshape(1, -1), self.centroids)[0]
        nearest_cluster = np.argmax(similarities)
        
        self.hash_table[tuple(self.centroids[nearest_cluster])].append((normalized_vector, 1.0))
        
        # Perform regional k-means clustering
        self._regional_clustering(nearest_cluster)
        
        end_time = time.time()
        self.time_complexity.append(end_time - start_time)
        self.space_complexity.append(self._calculate_space_complexity())

    def _regional_clustering(self, cluster_index):
        cluster_vectors = [v for v, _ in self.hash_table[tuple(self.centroids[cluster_index])]]
        if len(cluster_vectors) > max(10, int(0.1 * len(self.centroids))):  # Adjust these thresholds as needed
            X = np.array(cluster_vectors)
            optimal_params = self.fit_cluster(X)
            n_clusters = optimal_params['n_clusters']
            
            if n_clusters > 1:
                new_centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]
                new_centroids = new_centroids / np.linalg.norm(new_centroids, axis=1, keepdims=True)
                
                for iteration in range(self.max_iter):
                    clusters = self._assign_clusters(X, new_centroids)
                    updated_centroids = self._update_centroids(X, clusters, new_centroids)
                    
                    if np.allclose(new_centroids, updated_centroids, atol=1e-5):
                        break
                    
                    new_centroids = updated_centroids
                
                # Update hash table with new clusters
                del self.hash_table[tuple(self.centroids[cluster_index])]
                for i, centroid in enumerate(new_centroids):
                    self.hash_table[tuple(centroid)] = [(X[j], 1.0) for j in np.where(clusters == i)[0]]
                
                # Update centroids
                self.centroids = np.vstack((np.delete(self.centroids, cluster_index, axis=0), new_centroids))
            else:
                # If only one cluster is formed, keep the existing centroid
                print("Only one cluster formed during regional clustering. Keeping existing centroid.")
        else:
            print(f"Not enough vectors ({len(cluster_vectors)}) for regional clustering. Skipping.")

    def search(self, query_vector, top_k=5):
        start_time = time.time()
        
        normalized_query = query_vector / np.linalg.norm(query_vector)
        similarities = cosine_similarity(normalized_query.reshape(1, -1), self.centroids)[0]
        top_centroids = np.argsort(similarities)[::-1][:top_k]
        
        most_similar_vectors = []
        for centroid_idx in top_centroids:
            centroid = self.centroids[centroid_idx]
            vectors = [v for v, _ in self.hash_table[tuple(centroid)]]
            if vectors:
                vector_similarities = cosine_similarity(normalized_query.reshape(1, -1), vectors)[0]
                most_similar_idx = np.argmax(vector_similarities)
                most_similar_vectors.append((vectors[most_similar_idx], vector_similarities[most_similar_idx]))
                
                # Boost attention
                self.hash_table[tuple(centroid)][most_similar_idx] = (
                    vectors[most_similar_idx],
                    min(self.hash_table[tuple(centroid)][most_similar_idx][1] * self.boost_rate, 1.0)
                )
        
        most_similar_vectors.sort(key=lambda x: x[1], reverse=True)
        
        end_time = time.time()
        self.time_complexity.append(end_time - start_time)
        self.space_complexity.append(self._calculate_space_complexity())
        
        return most_similar_vectors[:top_k]

    def maintain_memory(self):
        for centroid, vectors in self.hash_table.items():
            updated_vectors = []
            for vector, attention in vectors:
                new_attention = attention * self.decay_rate
                if new_attention > self.attention_threshold:
                    updated_vectors.append((vector, new_attention))
            self.hash_table[centroid] = updated_vectors
    
    def _calculate_space_complexity(self):
        return sum(len(vectors) for vectors in self.hash_table.values()) * self.centroids.shape[1]

    def analyze_time_complexity(self, operation, *args, **kwargs):
        start_time = time.perf_counter()
        result = getattr(self, operation)(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Time complexity of {operation}: {execution_time:.6f} seconds")
    
        # Estimate Big O notation
        n = len(self.hash_table)  # number of clusters
        m = sum(len(vectors) for vectors in self.hash_table.values())  # total number of vectors
        d = self.centroids.shape[1]  # dimension of vectors
    
        if operation == 'add_vector':
            print(f"Estimated time complexity: O(n*d) where n={n} (clusters) and d={d} (dimensions)")
        elif operation in ['search']:
            print(f"Estimated time complexity: O(n*d + k*m*d) where n={n} (clusters), m={m} (vectors), d={d} (dimensions), and k is the number of top centroids")
        elif operation == 'maintain_memory':
            print(f"Estimated time complexity: O(m) where m={m} (total vectors)")
    
        return result

    def analyze_space_complexity(self):
        total_size = sum(sys.getsizeof(centroid) + sum(sys.getsizeof(v) + sys.getsizeof(a) for v, a in vectors) 
                         for centroid, vectors in self.hash_table.items())
        total_size += sys.getsizeof(self.centroids)
        
        total_size_mb = total_size / (1024 * 1024)  # Convert bytes to MB
        print(f"Space complexity: {total_size_mb:.2f} MB")
        
        # Estimate Big O notation
        n = len(self.hash_table)  # number of clusters
        m = sum(len(vectors) for vectors in self.hash_table.values())  # total number of vectors
        d = self.centroids.shape[1]  # dimension of vectors
        print(f"Estimated space complexity: O(n*d + m*d) where n={n} (clusters), m={m} (vectors), and d={d} (dimensions)")
        
        return total_size_mb

    def analyze_operations(self, n_operations=1000):
        print(f"\nAnalyzing {n_operations} operations:")
        
        # Analyze add_vector
        add_times = []
        for _ in range(n_operations):
            vector = np.random.rand(self.centroids.shape[1])
            start_time = time.perf_counter()
            self.add_vector(vector)
            end_time = time.perf_counter()
            add_times.append(end_time - start_time)
        print(f"Average add_vector time: {np.mean(add_times):.6f} seconds")
        print(f"Estimated time complexity: O(n*d) per operation")
        
        # Analyze search
        search_times = []
        for _ in range(n_operations):
            query = np.random.rand(self.centroids.shape[1])
            start_time = time.perf_counter()
            self.search(query)
            end_time = time.perf_counter()
            search_times.append(end_time - start_time)
        print(f"Average search time: {np.mean(search_times):.6f} seconds")
        print(f"Estimated time complexity: O(n*d + k*m*d) per operation")
        
        # Analyze maintain_memory
        start_time = time.perf_counter()
        self.maintain_memory()
        end_time = time.perf_counter()
        print(f"maintain_memory time: {end_time - start_time:.6f} seconds")
        print(f"Estimated time complexity: O(m)")
        
        # Analyze space complexity
        self.analyze_space_complexity()

# Usage example
np.random.seed(42)
sample_data = np.random.rand(60000, 768)  # 10000 vectors of dimension 256

##stm = ShortTermMemory(max_iter=100, random_state=42)
#stm.fit(sample_data)
print("Fitting completed successfully")

# Add some vectors
for _ in range(6000):
    new_vector = np.random.rand(256)
    #stm.add_vector(new_vector)

# Compare regular search and threaded search
query_vector = np.random.rand(256)

print("\nRegular Search:")
#stm.analyze_time_complexity('search', query_vector, top_k=5)

# Perform bulk analysis
#stm.analyze_operations(n_operations=100)