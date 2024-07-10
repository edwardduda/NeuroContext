## NeuroContext - Memory storage for Large Language Model context.
This project focuses on creating a self-organizing vector embedding space used to improve traversal and Retrieval-Augmented Generation to improve the accuracy, reliability, and quality of Large Langauge Models. 

# Background

Vector embeddings are a way of representing discrete data in a continuous vector space of n dimensions. This process transforms data into a numerical format that preserves the relationships and structure within the data. By mapping data to a vector embedding space, these embeddings allow AI algorithms, such as neural networks, to interpret and utilize the underlying patterns and meanings within the data.

In transformer neural networks, the primary architecture for modern-day LLMs, the encoder is responsible for taking input data (such as text) and processing it through multiple layers of self-attention and feed-forward neural networks. Through this process, the encoder creates high-dimensional vector representations that capture the semantic meaning and contextual information of the input data. These embeddings can then be used for various downstream tasks, such as language translation, sentiment analysis, and more. However, it is important to note that not all transformer-based neural networks use encoders; vector embeddings can be generated in other ways as well.

# Cosine similarity and its role in context
There are many ways to determine the congruency of a pair of vectors, one of them being cosine similarity. Cosine similarity is used to determine how similar the direction of the vectors is. The closer this value is to 1, the more similar their direction. This measure is crucial for assessing the contextual relevance between different pieces of data within the embedding space. Words with more association have a higher similarity.

![Cosine Similarity applied.](https://github.com/edwardduda/NeuroContext/blob/174930272101e329650ac97317969644adc38f3a/1_jptD3Rur8gUOftw-XHrezQ-2342057285.png)

# K-means Clustering

K-means clustering is an unsupervised machine learning algorithm used to partition a dataset into k distinct, non-overlapping groups or clusters. The goal is to group similar data points together while ensuring that the clusters are as distinct as possible from one another. Here’s a high-level overview of how it works:

Initialization: Randomly select k initial cluster centroids (means).
Assignment: Assign each data point to the nearest cluster centroid based on Euclidean distance.
Update: Recalculate the centroids as the mean of all data points assigned to each cluster.
Iterate: Repeat the assignment and update steps until the centroids no longer change significantly or a predefined number of iterations is reached.
K-means is efficient and easy to implement, but it has some limitations, such as sensitivity to the initial placement of centroids and difficulty in determining the optimal number of clusters (k). Despite these challenges, it is widely used for tasks like market segmentation, image compression, and pattern recognition.

![k-means clustering simplified in 3-dimensions.](https://github.com/edwardduda/NeuroContext/blob/174930272101e329650ac97317969644adc38f3a/1_yBT_wK_lGPuuVvKgAuAtYg.png)


