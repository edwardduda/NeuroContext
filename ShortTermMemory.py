import unittest
import torch
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.metrics import silhouette_score
from bayes_opt import BayesianOptimization

class ShortTermMemory:
    def __init__(self, X, num_clusters):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = 92
        self.centroids = {}
        self.centroid_magnitudes = {}
        self.embedding_magnitudes = {}
        self.vector_embeddings = torch.tensor(X).to(self.device)
        self.clusters = num_clusters
        self.point_assignments = np.zeros(X.shape[0], dtype=int)

    def calc_embed_mag(self, v1):
        self.embedding_magnitudes[tuple(v1.cpu().numpy())] = torch.norm(v1).item()

    def calc_cent_mag(self, v1):
        self.centroid_magnitudes[tuple(v1.cpu().numpy())] = torch.norm(v1).item()

    def initialize_centroids(self):
        torch.manual_seed(self.seed)
        centroids = []
        initial_index = torch.randint(0, self.vector_embeddings.shape[0], (1,)).item()
        centroids.append(self.vector_embeddings[initial_index])

        for _ in range(1, self.clusters):
            distances = torch.tensor([min([torch.norm(vec - c) for c in centroids]) for vec in self.vector_embeddings])
            probabilities = distances / distances.sum()
            cumulative_probabilities = torch.cumsum(probabilities, dim=0)
            r = torch.rand(1).item()

            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    centroids.append(self.vector_embeddings[j])
                    break

        self.centroids = {i: centroid for i, centroid in enumerate(centroids)}
        for i in range(self.clusters):
            self.calc_cent_mag(self.centroids[i])

    def cosine_similarity(self, v1, qv):
        v1_tuple = tuple(v1.cpu().numpy())
        qv_tuple = tuple(qv.cpu().numpy())

        if v1_tuple not in self.centroid_magnitudes:
            self.calc_cent_mag(v1)
        if qv_tuple not in self.embedding_magnitudes:
            self.calc_embed_mag(qv)

        dot_product = torch.dot(v1, qv).item()
        v1_mag = self.centroid_magnitudes[v1_tuple]
        qv_mag = self.embedding_magnitudes[qv_tuple]
        return dot_product / (v1_mag * qv_mag)

    def fit(self):
        new_centroids = {i: [] for i in range(self.clusters)}

        for i in range(self.vector_embeddings.shape[0]):
            max_similarity = -1.0
            best_centroid = None
            for j in range(self.clusters):
                similarity = self.cosine_similarity(self.centroids[j], self.vector_embeddings[i])
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_centroid = j
            new_centroids[best_centroid].append(self.vector_embeddings[i])
            self.point_assignments[i] = best_centroid

        for i in range(self.clusters):
            if new_centroids[i]:
                self.centroids[i] = torch.mean(torch.stack(new_centroids[i]), dim=0)
                self.calc_cent_mag(self.centroids[i])
            else:
                print(f"Cluster {i} is empty after fitting")

        # Debugging: Print the sizes of each cluster
        for i in range(self.clusters):
            print(f"Cluster {i} size: {len(new_centroids[i])}")

    def loss_function(self):
        silhouette = silhouette_score(self.vector_embeddings.cpu().numpy(), self.point_assignments, metric='cosine')
        return 1 - silhouette

    def target_function(self, num_clusters):
        self.clusters = int(num_clusters)
        self.initialize_centroids()
        self.fit()
        return self.loss_function()

    def optimize(self):
        p_bounds = {
            "num_clusters": (2, 98)
        }

        self.optimizer = BayesianOptimization(
            f=self.target_function,
            pbounds=p_bounds,
        )

        self.optimizer.maximize(init_points=3, n_iter=200)

        best_params = self.optimizer.max['params']
        print(f"Best parameters: {best_params}")
        print(f"Best value: {-self.optimizer.max['target']}")

    def print_clusters(self, sentences):
        cluster_sentences = {i: [] for i in range(self.clusters)}
        for idx, assignment in enumerate(self.point_assignments):
            cluster_sentences[assignment].append(sentences[idx])

        for cluster_id, sents in cluster_sentences.items():
            print(f"Cluster {cluster_id}:")
            for sent in sents:
                print(f"  - {sent}")
            print()

class TestShortTermMemory(unittest.TestCase):
    def setUp(self):
        # Use GPT-2 to create actual word embeddings
        model_name = 'gpt2'
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Add padding token
        self.model = GPT2Model.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')

        self.sentences = [
            "Apples are fruits that keep the doctor away.",
            "Georgia is known for its peaches",
            "Orange county is originally known for its orange groves",
            "The greater bay area in California is known as Silicon Valley",
            "Many schools and tech companies are in Silicon Valley"
        ]

        # Generate embeddings
        self.embeddings = self.generate_embeddings(self.sentences)
        self.num_clusters = 2
        self.stm = ShortTermMemory(self.embeddings, self.num_clusters)

    def generate_embeddings(self, sentences):
        embeddings = []
        for sentence in sentences:
            inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            last_hidden_states = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_size)
            sentence_embedding = torch.mean(last_hidden_states, dim=0).cpu().numpy()  # Mean pooling
            embeddings.append(sentence_embedding)
        return np.array(embeddings)

    def test_initialize_centroids(self):
        self.stm.initialize_centroids()
        self.assertEqual(len(self.stm.centroids), self.num_clusters)
        for i in range(self.num_clusters):
            self.assertTrue(isinstance(self.stm.centroids[i], torch.Tensor))

    def test_calc_embed_mag(self):
        v = torch.tensor([1.0, 2.0, 3.0]).to(self.stm.device)
        self.stm.calc_embed_mag(v)
        v_tuple = tuple(v.cpu().numpy())
        self.assertIn(v_tuple, self.stm.embedding_magnitudes)
        self.assertAlmostEqual(self.stm.embedding_magnitudes[v_tuple], torch.norm(v).item())

    def test_calc_cent_mag(self):
        v = torch.tensor([1.0, 2.0, 3.0]).to(self.stm.device)
        self.stm.calc_cent_mag(v)
        v_tuple = tuple(v.cpu().numpy())
        self.assertIn(v_tuple, self.stm.centroid_magnitudes)
        self.assertAlmostEqual(self.stm.centroid_magnitudes[v_tuple], torch.norm(v).item())

    def test_cosine_similarity(self):
        v1 = torch.tensor([1.0, 0.0, 0.0]).to(self.stm.device)
        qv = torch.tensor([0.0, 1.0, 0.0]).to(self.stm.device)
        similarity = self.stm.cosine_similarity(v1, qv)
        self.assertAlmostEqual(similarity, 0.0)

    def test_fit(self):
        self.stm.initialize_centroids()
        self.stm.fit()
        self.assertEqual(len(self.stm.centroids), self.num_clusters)
        self.assertEqual(len(self.stm.point_assignments), self.stm.vector_embeddings.shape[0])
        self.stm.print_clusters(self.sentences)

    def test_loss_function(self):
        self.stm.initialize_centroids()
        self.stm.fit()
        loss = self.stm.loss_function()
        self.assertTrue(0 <= loss <= 1)
'''
    def test_optimize(self):
        self.stm.optimize()
        best_params = self.stm.optimizer.max['params']
        self.assertIn('num_clusters', best_params)
        self.assertTrue(2 <= best_params['num_clusters'] <= 98)
'''
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
