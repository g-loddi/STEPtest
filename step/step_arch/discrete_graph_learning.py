# Discrete Graph Learning
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from basicts.utils import load_pkl

from .similarity import batch_cosine_similarity, batch_dot_similarity


def sample_gumbel(shape, eps=1e-20, device=None):
    """
    To sample the noise from a Gumbel distribution, the code below use the Inverse Transform Sampling method
    (see also https://en.wikipedia.org/wiki/Inverse_transform_sampling).

    First, we take samples from the uniform distribution in the range [0,1): these represent probabilities.
    Then, we use the Gumbel inversed CDF and feed to it the aforementioned probabilities to get the values of a random variable
    following the Gumbel distribution.
    """

    # Generate a tensor whose values are drawn from the uniform distribution in [0,1).
    # These represent the probabilities used to draw samples from the Gumbel distribution.
    uniform = torch.rand(shape).to(device)

    # Generate the actual Gumbel noise using its Inversed CDF and the probabilities generated from a uniform distribution. 
    # NOTE 1: "eps" serves to avoid computing logaritms of zero or very small values, which might lead to -inf.
    # NOTE 2: the use of 'torch.autograd.Variable' is useless and can be removed.
    # NOTE 3: Recall that the Gumbel CDF is:
    #
    #         F(x) = exp(-exp(-\frac{x - \mu}{\beta})) = p,
    #
    #         where "p" is the probability that a random variable will take a value <= x. 
    #         Recall that the inverse of a CDF, or F^{-1}(p), returns the smallest "x" such that F(x) = p.
    #         In the case of Gumbel's CDF, this can be derived by working on:
    #
    #         p = exp(-exp(-\frac{x - \mu}{\beta})) \in [0,1)
    #    
    #         and with some algebraic manipulations using the natural logarithm we arrive at:
    # 
    #         x = \mu - \beta * ln(-ln(p))
    # 
    #         In the code it is assumed that \mu = 0 and \beta = 1, so the eq. simplifies to: x = -ln(-ln(p)),
    #         and thus the inverse CDF is: F^{-1}(p) = -ln(-ln(p))
    # 
    #         Then, by feeding the "p" values from "uniform" in the inversed CDF give us the values of a random variable following the Gumbel distribution.
    return -torch.autograd.Variable(torch.log(-torch.log(uniform + eps) + eps))


def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    
    # Generates Gumbel noise of the same shape as the logits.
    sample = sample_gumbel(logits.size(), eps=eps, device=logits.device)
    # print(f"DEBUG FRA, discrete_graph_learning.py.gumbel_softmax_sample() => logits shape: {logits.shape}")
    # print(f"DEBUG FRA, discrete_graph_learning.py.gumbel_softmax_sample() => sample shape: {sample.shape}")
    
    # Use the Gumbel noise to perturb the logits. This ensures that we are actually sampling from the prob. dist.
    # (i.e., the Bernoulli one) represented by the logits, and that the sampling process is differentiable.
    # NOTE: This works since the final normalized probabilities computed by the softmax are determined by the distances between the logits
    # rather than their absolute values. For example, the logits [2.0, 1.9] and [2000.0, 1999.0] yield the same normalized
    # probabilities, while [2.0, 1.9] and [2000.0, 1900.0] don't. This explains why implementing sampling by perturbing the logits with noise works.
    # Moreover, the Gumbel noise is the best one to preserve the original categorical distribution represented by the logits.
    y = logits + sample
    
    # The samples are then passed through a softmax function, scaled by the temperature.
    # This returns a tensor of shape "[batch_size, num_edges, 2]", where each pair of values represents
    # the probabilities of the edge being absent or present.
    # NOTE: the smaller the temperature, the closer we get to the categorical distribution. The larger, the closer to the uniform distribution.
    #       Best temperature values are typically around 0.5 (as is the case in the paper), but other approaches might perform annealing to find
    #       even better values for some considered scenario. 
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    The use of Gumbel noise in the context of the Gumbel-Softmax trick is a way 
    to create a differentiable approximation of discrete random variables, 
    which is the case with the Bernoulli distribution, and is essential for backpropagation
    in neural networks. Moreover, introducing sampling allow the model to better explore the 
    solution space when training.

    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y

    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
    """

    y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        
        # This version of the max function finds the maximum values and their indexes along the last dimension.
        # Since we are interested in the indexes, we are using max as an argmax.
        _, k = y_soft.data.max(-1)
        

        # Creates a tensor of zeros with the same shape as logits, then uses scatter_ to create a one-hot 
        # encoded tensor based on the indices from the argmax operation.
        y_hard = torch.zeros(*shape).to(logits.device)


        # NOTE: Tensor.scatter_(dim, index, src, *, reduce=None): writes all values from the 
        # tensor src=k into self at the indices specified in the index "k" tensor. For each 
        # value in k, its output index is specified by its index in k for dimension != dim 
        # and by the corresponding value in index for dimension = dim.
        # NOTE 2: k has shape [batch_size, num_edges], while its view has shape [batch_size, num_edges, 1].
        # NOTE 3: the reshape of "k" could be written in more simple terms as "k.unsqueeze(-1)".
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)


        # Straight-Through Estimator: Uses the straight-through estimator to create a tensor that is one-hot during
        # the forward pass but has gradients from the soft sample during backpropagation. Overall, this ensures that the sampling
        # process is differentiable when doing backprop, while allowing to make hard decisions given a probability distribution when
        # doing the forward pass.
        # NOTE: Here, y_hard is the hard one-hot sample, and y_soft is the continuous soft sample. The .detach() operation prevents gradients
        # from flowing through y_hard, while the addition of "y_soft" at the end ensures that the gradient during backpropagation is taken from
        # the soft sample.
        # NOTE 2: instead of "torch.autograd.Variable(y_hard - y_soft.data)", which effectively creates a new tensor detached from the computation
        #         graph, we can simply write "(y_hard - y_soft.data).detach()"
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft # equivalent to the more modern "(y_hard - y_soft.data).detach() + y_soft"
    else:
        y = y_soft
    
    return y


class DiscreteGraphLearning(nn.Module):
    """Dynamic graph learning module."""

    def __init__(self, dataset_name, k, input_seq_len, output_seq_len):
        super().__init__()

        self.k = k          # the "k" of knn graph


        # NOTE: the authors added these dictionaries with constants from which the appropriate values are chosen.
        self.num_nodes = {"METR-LA": 207, "PEMS04": 307, "PEMS03": 358, "PEMS-BAY": 325, "PEMS07": 883, "PEMS08": 170}[dataset_name]
        self.train_length = {"METR-LA": 23990, "PEMS04": 13599, "PEMS03": 15303, "PEMS07": 16513, "PEMS-BAY": 36482, "PEMS08": 14284}[dataset_name]
        
        # Here we read the original time series data (2d array) generated by the nodes.
        self.node_feats = torch.from_numpy(load_pkl("datasets/" + dataset_name + "/data_in{0}_out{1}.pkl".format(input_seq_len, output_seq_len))["processed_data"]).float()[:self.train_length, :, 0]


        # Convolutional + other layers for global feature extraction (this is G^i in eq.2)
        ## for the dimension, see https://github.com/zezhishao/STEP/issues/1#issuecomment-1191640023
        # NOTE: this part is related to Section "3.2 - The Forecasting Stage", eq.2, of the paper.
        self.dim_fc = {"METR-LA": 383552, "PEMS04": 217296, "PEMS03": 244560, "PEMS07": 263920, "PEMS-BAY": 583424, "PEMS08": 228256}[dataset_name]
        self.embedding_dim = 100
        ## network structure
        self.conv1 = torch.nn.Conv1d(1, 8, 10, stride=1)  # .to(device)
        self.conv2 = torch.nn.Conv1d(8, 16, 10, stride=1)  # .to(device)
        self.fc = torch.nn.Linear(self.dim_fc, self.embedding_dim)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(self.embedding_dim)

        # NOTE: dim_fc_mean and fc_mean are not used, see the authors' comment in the forward method.
        ## for the dimension, see https://github.com/zezhishao/STEP/issues/1#issuecomment-1191640023
        self.dim_fc_mean = {"METR-LA": 16128, "PEMS-BAY": 16128, "PEMS03": 16128 * 2, "PEMS04": 16128 * 2, "PEMS07": 16128, "PEMS08": 16128 * 2}[dataset_name]
        self.fc_mean = nn.Linear(self.dim_fc_mean, 100)


        # discrete graph learning part (related to eq.2 in the paper)
        self.fc_cat = nn.Linear(self.embedding_dim, 2) # First FC layer, applied after concat.
        self.fc_out = nn.Linear((self.embedding_dim) * 2, self.embedding_dim)
        self.dropout = nn.Dropout(0.5)


        def encode_one_hot(labels):
        # reference code https://github.com/chaoshangcs/GTS/blob/8ed45ff1476639f78c382ff09ecca8e60523e7ce/model/pytorch/model.py#L149
            classes = set(labels) # This is equivalent to computing "unique" over a dataframe.
            # From the identity array, select the vector representing the one-hot encoding for a given node.
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
            # The line below retrieves the appropriate vector from the the dict "classes_dict" for each node index in "labels".
            # This produces a 2D array of shape "[len(labels), len(classes_dict[0])]".
            labels_one_hot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
            return labels_one_hot


        # The code below generates one-hot representations of the receiver and sender nodes.
        # NOTE: when we provide only the "condition" argument to np.where, where "condition" is a boolean 2D array, then np.where scans "condition"
        #       row-wise and returns a tuple containing two 1D arrays. The first 1D array, which is then used to build "self.rec_rec", contains the 
        #       row-indices of the non-zero elements within "condition". These are actually the sender nodes (i.e., the ones from which edges originate),
        #       but the authors appear to have swapped their meaning.
        #  
        #       The second 1D array contains the column-indices of the non-zero elements within "condition". These are actually the receiver nodes 
        #       -- again, the authors appear to have swapped their meaning.
        #       
        #       Both 1D arrays have length equal to the number of elements in "condition", i.e., num_nodes^2.
        #       Probably, the fact that the authors swapped their meaning should not have repercussions -- see code in the forward method.
        #
        #       From these two 1D arrays, we create two 2D tensor, one for receiver nodes and one for sender nodes, containing the one-hot encodings
        #       of the nodes rather than their indexes. As a result, both 2D tensors have shape "[num_nodes^2, num_nodes]", where "num_nodes" is due
        #       to the one-hot encodings.
        self.rel_rec = torch.FloatTensor(np.array(encode_one_hot(np.where(np.ones((self.num_nodes, self.num_nodes)))[0]), dtype=np.float32))
        self.rel_send = torch.FloatTensor(np.array(encode_one_hot(np.where(np.ones((self.num_nodes, self.num_nodes)))[1]), dtype=np.float32))


    def get_k_nn_neighbor(self, data, k=11*207, metric="cosine"):
        """
        data: tensor B, N, D
        metric: cosine or dot
        """

        if metric == "cosine":
            batch_sim = batch_cosine_similarity(data, data)
        elif metric == "dot":
            batch_sim = batch_dot_similarity(data, data)    # B, N, N
        else:
            assert False, "unknown metric"
        batch_size, num_nodes, _ = batch_sim.shape
        adj = batch_sim.view(batch_size, num_nodes*num_nodes)
        res = torch.zeros_like(adj)
        top_k, indices = torch.topk(adj, k, dim=-1)
        res.scatter_(-1, indices, top_k)
        adj = torch.where(res != 0, 1.0, 0.0).detach().clone()
        adj = adj.view(batch_size, num_nodes, num_nodes)
        adj.requires_grad = False
        return adj


    def forward(self, long_term_history, tsformer):
        """Learning discrete graph structure based on TSFormer.

        Args:
            long_term_history (torch.Tensor): very long-term historical MTS with shape [B, P * L, N, C], which is used in the TSFormer.
                                                P is the number of segments (patches), and L is the length of segments (patches).
            tsformer (nn.Module): the pre-trained TSFormer.

        Returns:
            torch.Tensor: Bernoulli parameter (unnormalized) of each edge of the learned dependency graph. Shape: [B, N * N, 2].
            torch.Tensor: the output of TSFormer with shape [B, N, P, d].
            torch.Tensor: the kNN graph with shape [B, N, N], which is used to guide the training of the dependency graph.
            torch.Tensor: the sampled graph with shape [B, N, N].
        """

        device = long_term_history.device
        batch_size, _, num_nodes, _ = long_term_history.shape
        

        # 1 - generate global feature
        print(f"DEBUG FRA, discrete_graph_learning.forward() => self.node_feats shape: {self.node_feats.shape}")
        # 1.1 - Initially, self.node_feats has shape [# timesteps, N]. From it, we generate global_feat, which has shape [N, 1, # timesteps].
        #       This is needed to process the data into the 1D convolutional layers, which expect a tensor of shape [batch = N, C_in, length].
        global_feat = self.node_feats.to(device).transpose(1, 0).view(num_nodes, 1, -1)
        print(f"DEBUG FRA, discrete_graph_learning.forward() => global_feat shape / 1: {global_feat.shape}")
        # 1.2 - Apply the first Conv1D layer. This layer expect 1 channel (which is the # features per timestep we have in the nodes'
        #       univariate timeseries), and applies 8 filters, thus yielding 8 features (c_out=8) per timestep. Finally, the Conv1D layer uses a kernel
        #       size of 10, padding=0, dilation=1, and stride=1, hence the new length of the timeseries can be simply computed as "length - 9".
        #       This yields a new tensor with shape [N, C_out=8, new_length1_ts=length-9]
        global_feat = self.bn1(F.relu(self.conv1(global_feat)))
        print(f"DEBUG FRA, discrete_graph_learning.forward() => global_feat shape / 2: {global_feat.shape}")
        # 1.3 - Apply the second Conv1D layer. This layer expect 8 channels (the ones from the first Conv1D layer), and applies 16 filters, 
        #       thus yielding 16 features (c_out=16) per timestep. Finally, this second Conv1D layer uses a kernel size of 10, padding=0, 
        #       dilation=1, and stride=1, hence the new length of the timeseries can be simply computed as "length - 9".
        #       This yields a new tensor with shape [N, C_out=16, new_length2_ts=new_length1_ts - 9]
        global_feat = self.bn2(F.relu(self.conv2(global_feat)))
        print(f"DEBUG FRA, discrete_graph_learning.forward() => global_feat shape / 3: {global_feat.shape}")
        # 1.4 - Now flatten global_feat. In practice, for every node we are stacking horizontally the 16 channels generated by the last Conv1D layer.
        #       This explains the constants stored in 'self.dim_fc'.
        global_feat = global_feat.view(num_nodes, -1)
        print(f"DEBUG FRA, discrete_graph_learning.forward() => global_feat shape / 4: {global_feat.shape}")
        # 1.5 - Then, use a linear layer + RELU + BN to project the above-mentioned mega vectors in a 100-dimensional space.
        global_feat = F.relu(self.fc(global_feat))
        global_feat = self.bn3(global_feat)
        print(f"DEBUG FRA, discrete_graph_learning.forward() => global_feat shape / 5: {global_feat.shape}")
        # 1.5 - Finally, add a singleton first dimension, and replicate the resulting vector "batch_size" times row-wise.
        #       Doing this enables to make global_feat available for every element in a batch.
        global_feat = global_feat.unsqueeze(0).expand(batch_size, num_nodes, -1)                     # Gi in Eq. (2)
        print(f"DEBUG FRA, discrete_graph_learning.forward() => global_feat shape / 6: {global_feat.shape}")


        # Compute the embeddings of the multivariate timeseries with TSFormer encoder.
        # (IGNORE this comment) generate dynamic feature based on TSFormer
        # NOTE: the line below preserves (via the ellipsis) all the dimensions except the last one, in which we only select the first feature.  
        hidden_states = tsformer(long_term_history[..., [0]])
        # print(f"DEBUG FRA, discrete_graph_learning.forward() => long_term_history shape: {long_term_history.shape}")
        # print(f"DEBUG FRA, discrete_graph_learning.forward() => hidden_states shape: {hidden_states.shape}")
        
        # NOTE (from the AUTHORS): the dynamic feature has now been removed, as we found that it could lead to instability in the learning
        #                            of the underlying graph structure.
        # NOTE 2: So, it seems that the authors have "disabled" the first term of the sum in the second equation of eq.2, i.e.,
        #         we simply have Z^i = G^i.
        # dynamic_feat = F.relu(self.fc_mean(hidden_states.reshape(batch_size, num_nodes, -1)))     # relu(FC(Hi)) in Eq. (2)


        # time series feature (it's just an alias for the "global_feat" tensor)
        node_feat = global_feat


        ### Learning discrete graph structure ###
        print(f"DEBUG FRA, discrete_graph_learning.forward() => self.rel_rec shape: {self.rel_rec.shape}")
        print(f"DEBUG FRA, discrete_graph_learning.forward() => self.rel_send shape: {self.rel_send.shape}")
        print(f"DEBUG FRA, discrete_graph_learning.forward() => node_feat shape: {node_feat.shape}")

        # The overall objective of the two matmuls below, which perform a "broadcasted matrix x batched matrix" multiplication
        # is to create two matrices with shape "[num_nodes^2, dim(global_feat)]", namely "receivers" and "senders". Note that "num_nodes^2"
        # is the number of edges in the graph if there exist an edge for every pair of nodes.
        #
        # Recall that "rel_rec" contains the one-hot encodings of the nodes that are on the receiving side of an edge, while "rel_send"
        # contains the one-hot encodings of nodes that are on the originating side of an edge. Both have shape [num_nodes^2, num_nodes].
        # 
        # Furthermore, recall that "node_feat" = "global_feat" contains the global features of the windows of the timeseries in a batch,
        # and it has shape [batches, num_nodes, dim(global_feat) = 100].
        # 
        # So, the act of performing a batch multiplication between, e.g., "self.rel_rec" and "node_feat" yields a 3D tensor of shape
        # [batches, num_nodes^2, dim(global_feat) = 100], where each submatrix [num_nodes^2, dim(global_feat) = 100] contains the global features
        # computed over the windows of the time series of receiver nodes. In other words, the one-hot encodings serve to select the global features
        # associated to the various receiver nodes.
        #
        # Analogous line of reasoning applies to "senders"
        receivers = torch.matmul(self.rel_rec.to(node_feat.device), node_feat) # This matrix contains the various Z^is = G^is used in eq.2
        senders = torch.matmul(self.rel_send.to(node_feat.device), node_feat) # This matrix contains the various Z^js = G^js used in eq.2
        print(f"DEBUG FRA, discrete_graph_learning.forward() => receivers shape: {receivers.shape}")
        print(f"DEBUG FRA, discrete_graph_learning.forward() => senders shape: {senders.shape}")
        
        # Computing the Bernoulli parameter (unnormalized) Theta in Eq. (2)
        edge_feat = torch.cat([senders, receivers], dim=-1) # We concatenate senders and receivers on the dimension of the global feature embeddings.
                                                            # In other words, for each edge, we concat the global feature embeddings of the sender
                                                            # and receiver, thus yielding an embedding double the size of the former two.
        print(f"DEBUG FRA, discrete_graph_learning.forward() => edge_feat shape: {edge_feat.shape}")
        edge_feat = torch.relu(self.fc_out(edge_feat)) # FC (halves the second dim of edge_feat) + RELU
        print(f"DEBUG FRA, discrete_graph_learning.forward() => edge_feat shape / 2: {edge_feat.shape}")
        bernoulli_unnorm = self.fc_cat(edge_feat) # Second final FC: this reduces the second dimension of edge_feat to 2: it yields 
                                                  # the unnormalized /theta. Shape: "[batch, num_nodes^2, 2]". 
        print(f"DEBUG FRA, discrete_graph_learning.forward() => bernoulli_unnorm shape: {bernoulli_unnorm.shape}")
        
        
        # NOTE: given an element in the batch, we have a submatrix with shape "[num_edges, 2]". This means that for each edge we have two logits,
        #       representing, respectively, the probability that an edge is present (first logit) or absent (second logit).  


        ## Differentiable sampling via Gumbel-Softmax in Eq. (4): this trick makes the sampling process differentiable.
        # NOTE: this generate samples from the logits in bernoulli_unnorm in the form of one-hot encodings (vectors with 2 els):
        #       if the first position is 1 then an edge exists, if the second position is one then the edge does not exist.
        #       This yields a tensor with shape [batch_size, num_edges, 2].
        sampled_adj = gumbel_softmax(bernoulli_unnorm, temperature=0.5, hard=True)
        print(f"DEBUG FRA, discrete_graph_learning.forward() => sampled_adj shape: {sampled_adj.shape}")


        # NOTE: given a batch element and an edge, the slicing "sampled_adj[..., 0]" preserves only the first element in the one-hot encoding,
        #        i.e., we are interested to know just if an edge is present. This yields a tensor with shape [num_batches, num_edges], i.e., each
        #        batch element contains a graph adjacency matrix.
        #        The final reshape then serves the purpose of making the flattened adjacency submatrix 2D, thus yielding a final tensor with shape
        #        [batch_size, num_nodes, num_nodes]. 
        # NOTE 2: the clone operation seems unnecesary, as the original sampled_adj is being overwritten.
        sampled_adj = sampled_adj[..., 0].clone().reshape(batch_size, num_nodes, -1)
        print(f"DEBUG FRA, discrete_graph_learning.forward() => sampled_adj shape / 2: {sampled_adj.shape}")
        

        # Removal of the self-loops within the adjacency matrices.
        # NOTE: first, eye creates an identity matrix, and the unsqueeze serves the purpose of creating a broadcastable identity matrix across
        #       the batch elements.
        #       The broadcastable identity matrix has thus shape [1, num_nodes, num_nodes] and is used to remove (via the masked_fill_) the 
        #       self-loops from the adjacency matrices (notice the zero value).
        mask = torch.eye(num_nodes, num_nodes).unsqueeze(0).bool().to(sampled_adj.device)
        sampled_adj.masked_fill_(mask, 0)


        # prior graph based on TSFormer (i.e., the A^a).
        adj_knn = self.get_k_nn_neighbor(hidden_states.reshape(batch_size, num_nodes, -1), k=self.k*self.num_nodes, metric="cosine")
        mask = torch.eye(num_nodes, num_nodes).unsqueeze(0).bool().to(adj_knn.device)
        adj_knn.masked_fill_(mask, 0)

        return bernoulli_unnorm, hidden_states, adj_knn, sampled_adj
