import math

import torch


def batch_cosine_similarity(x, y):
    '''
    Compute the cosine similarity between vectors. Recall that given two vectors a and b, their cosine similarity is given
    by /frac{a \cdot b}{||a|| ||b||}, i.e., the ratio between their dot product and the product of their L2 norms.

    NOTE: x and y have shapes [batch_size, num_nodes, dim_conc_emb]. Given the i-th batch in both, we want
          to compute the cosine similarity between the dim_embs in x with those in y. 
    '''

    # Compute the denominator, which is the product of the norms of the concatenated embeddings.
    # The resulting tensors will have shape [batch_size, num_nodes].
    # First, we calcuate the L2 norms.
    l2_x = torch.norm(x, dim=2, p=2) + 1e-7  # avoid 0, l2 norm
    l2_y = torch.norm(y, dim=2, p=2) + 1e-7  # avoid 0, l2 norm
    # print(f"DEBUG FRA, similarity.py.batch_cosine_similarity() => l2_x and l2_y shape: {l2_x.shape}")

    # We then prepare l2_x such that it has shape [batch_size, num_nodes, 1] and l2_y such that it has
    # shape [batch_size, 1, num_nodes]. Then we use matmul in order to perform a batch multiplication,
    # which yields a tensor with shape [batch_size, num_nodes, num_nodes] 
    l2_m = torch.matmul(l2_x.unsqueeze(dim=2), l2_y.unsqueeze(dim=2).transpose(1, 2))
    # print(f"DEBUG FRA, similarity.py.batch_cosine_similarity() => l2_m shape: {l2_m.shape}")
    
    # Compute the cosine similarity numerator, which is the dot product between the concat embeddings.
    # Note that we prepare "y" for batch multiplication, similarly to what we've done before for the denominator, so its shape 
    # will be [batch_size, dim_conc_emb, num_nodes]. Thus, "l2_z" will have shape [batch_size, num_nodes, num_nodes].
    l2_z = torch.matmul(x, y.transpose(1, 2))
    
    # Compute the final cosine similarity with the element-wise division below.
    cos_affnity = l2_z / l2_m
    adj = cos_affnity
    return adj

def batch_dot_similarity(x, y):
    QKT = torch.bmm(x, y.transpose(-1, -2)) / math.sqrt(x.shape[2])
    W = torch.softmax(QKT, dim=-1)
    return W
