from torch import nn


class PatchEmbedding(nn.Module):
    """Patchify time series."""

    def __init__(self, patch_size, in_channel, embed_dim, norm_layer):
        super().__init__()
        
        self.output_channel = embed_dim
        self.len_patch = patch_size             # the L
        self.input_channel = in_channel
        self.output_channel = embed_dim

        # 1D convolution applied over the time-series
        self.input_embedding = nn.Conv2d(in_channel,
                                         embed_dim,                         # Number of filters used in the Conv2D operator.
                                         kernel_size=(self.len_patch, 1),   # Each filter has size of a patch.
                                         stride=(self.len_patch, 1))        # We are sliding the filters with a jump equal to the length of each patch.
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()


    def forward(self, long_term_history):
        """
        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, 1, P * L],
                                                which is used in the TSFormer.
                                                P is the number of segments (patches).

        Returns:
            torch.Tensor: patchified time series with shape [B, N, d, P]
        """

        batch_size, num_nodes, num_feat, len_time_series = long_term_history.shape

        # The Conv2D operator expects a 4D input with shape (B, C, H, W). Generally speaking, with timeseries data, the height H corresponds to the 
        # temporal dimension, while the width W is not necessary, and so it becomes a singleton dimension.

        # "long_term_history" contains a multi-variate time-series with shape (B, N, C=1, P * L), where B is the size of batches,
        # N is the number of nodes (i.e., the features of the multivariate time-series), C=1 is the dimension holding the values 
        # and that has been added to make the multivariate timeseries processable by other NN layers, and P*L is the length of an 
        # historical window, with P representing the number of patches and L the length of each patch (12).
        
        # 1 - We first have to permute some dimensions in "long_term_history" to make it processable for Conv2D.
        # Each multivariate timeseries has shape (N, C=1, P*L), which becomes (N, P*L, C=1) 
        print(f"DEBUG FRA => PatchEmbedding.forward, shape long_term_history / 1: {long_term_history.shape}")
        
        # The unsqueeze serves the purpose of adding a singleton dimension at the end, making the tensor processable
        # by Conv2D down the line.
        long_term_history = long_term_history.unsqueeze(-1) # B, N, C=1, L, 1
        print(f"DEBUG FRA => PatchEmbedding.forward, shape long_term_history / 2: {long_term_history.shape}")
        
        # 2 - We reshape the tensor from (B, N, C=1, L, 1) to (B*N, C=1, L, 1); in other words, we stack vertically
        #     the multivariate timeseries within a batch. This is again needed, as Conv2D works on 4D tensors.
        long_term_history = long_term_history.reshape(batch_size*num_nodes, num_feat, len_time_series, 1)
        print(f"DEBUG FRA => PatchEmbedding.forward, shape long_term_history / 3: {long_term_history.shape}")
        
        # Here we apply the Conv1D operator (via the Conv2D one), in which we bring the original patches from a space
        # with dimension "L" to a higher space of dimension "d", thus yielding a tensor with shape: (B*N, d, L/P, 1)
        # NOTE: the authors here chose to apply the convolution operator independently on the univariate timeseries of individual
        #       sensors rather than the overall multivariate timeseries. More specifically, notice that the number of input channels
        #       used to instantiate Conv2D is 1 and that the shape of long_term_history is (B*N, C=1, L, 1),
        #       This is likely motivated by the fact that capturing features across the very few multivariate time series in a batch
        #       might not be as effective as applying it on the univariate timeseries of each node.
        print(f"DEBUG FRA => PatchEmbedding.forward, Conv2D number of input channels: {self.input_channel}")
        output = self.input_embedding(long_term_history)
        print(f"DEBUG FRA => PatchEmbedding.forward, shape output / 1: {output.shape}")
        
        # normalization layer (usually equivalent to the identity layer, see the TSFormer class)
        output = self.norm_layer(output)
        print(f"DEBUG FRA => PatchEmbedding.forward, shape output / 2: {output.shape}")
        
        # reshape the final tensor to from (B*N, C=1, L, 1) to (B, N, d, P): first, we eliminate the last dimension (squeeze), 
        # and then we reshape the tensor to undo the vertical stack (see view) previously done.
        output = output.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)
        print(f"DEBUG FRA => PatchEmbedding.forward, shape output / 3: {output.shape}")
        
        # Sanity check on the shape of the patchified output: we check that the last dimension is equal to the expected number of patches.
        assert output.shape[-1] == len_time_series / self.len_patch
        
        return output
