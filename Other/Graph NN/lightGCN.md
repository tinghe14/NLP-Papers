# LightGCN: simplifying and powering graph Convolution Network for Recommendation

1. adjacency matrix, user + item * user + item matrix
2. light version of Neural Graph Collaborative Filtering(NGCF)
3. removes the feature transformation matrix and non-linear activation function which prove to improve the preformance and loss function
4. reason: in the collabortive user-item matrix, each node uses one-hot encoding to represent. This reprensenter only should be used as identifier since it has no semantic meaning. When we have more transformation or non-linear caculuation, it won't help the performance
