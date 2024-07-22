import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, hidden_dim, dropout=0.1, max_len: int = 1000):
        '''
        Args

        hidden_dim: dimension of the embeddings of the patches computed by PatchEmbedding.
        max_len: maximum length of the sequence for which positional encodings are precomputed
        '''
        
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)

        # Creates a tensor containing learnable parameters representing the positional encoding applied to the patchified tokens.
        # Has shape (max_len, hidden_dim)
        self.position_embedding = nn.Parameter(torch.empty(max_len, hidden_dim), requires_grad=True)


    def forward(self, input_data, index=None, abs_idx=None):
        """Positional encoding

        Args:
            input_data (torch.tensor): input sequence with shape [B, N, P, d].
            index (list or None): add positional embedding by index. Typically omitted, and thus left to the default value, None

        Returns:
            torch.tensor: output sequence
        """

        batch_size, num_nodes, num_patches, num_feat = input_data.shape
        
        # We first reshape the input data from [B, N, P, d] to [B * N, P, d], we stack again vertically the multivariate patchified timeseries we
        # have in a batch.
        input_data = input_data.view(batch_size*num_nodes, num_patches, num_feat)
        
        # Select the learnable parameters in "self.position_embedding" according to the number of patches we have in the multivariate timeseries.
        # This yields a tensor "pe" with shape (P, d). Then, add a singleton dimension in the first position, yielding shape (1, P, d) 
        if index is None: # This is the typical case.
            pe = self.position_embedding[:input_data.size(1), :].unsqueeze(0)
        else:
            pe = self.position_embedding[index].unsqueeze(0)
        
        # Apply the positional encoding to the patchified multivariate timeseries, and then apply dropout. 
        # The positional encoding is applied on a per-node level.
        # NOTE: the broadcast is applied when computing the sum.
        input_data = input_data + pe
        input_data = self.dropout(input_data)
        
        # Bring back input_data to its original shape, i.e., [B, N, P, d].
        input_data = input_data.view(batch_size, num_nodes, num_patches, num_feat)
        return input_data
