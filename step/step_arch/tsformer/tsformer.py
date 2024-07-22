import torch
from torch import nn
from timm.models.vision_transformer import trunc_normal_

from .patch import PatchEmbedding
from .mask import MaskGenerator
from .positional_encoding import PositionalEncoding
from .transformer_layers import TransformerLayers


def unshuffle(shuffled_tokens):
    """This function reverses the shuffling of tokens, creating an unshuffle index."""

    dic = {}
    for k, v, in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index


class TSFormer(nn.Module):
    """An efficient unsupervised pre-training model for Time Series based on transFormer blocks. (TSFormer)"""

    def __init__(self, patch_size, in_channel, embed_dim, num_heads, mlp_ratio, dropout, num_token, mask_ratio, encoder_depth, decoder_depth, mode="pre-train"):
        super().__init__()        
        assert mode in ["pre-train", "forecasting"], "Error mode."

        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_token = num_token
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio

        # NOTE: the authors seem to consider only the 1st feature in the input data (?), i.e., the one coming from the raw data.
        self.selected_feature = 0

        # norm layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)


        ### encoder specifics ###

        # # patchify & embedding
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)
        
        # # positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout)
        
        # # masking
        # NOTE: num_token represents the number of patches in a time-series (e.g., 168 in the case of METR-LA).
        self.mask = MaskGenerator(num_token, mask_ratio)
        
        # encoder
        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)


        # ### decoder specifics ###

        # linear layer
        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        
        # # decoder
        self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)

        # # prediction (reconstruction) layer
        self.output_layer = nn.Linear(embed_dim, patch_size)
        self.initialize_weights()


    def initialize_weights(self):
        # Initialize the weights of the matrix representing the positional encoding
        nn.init.uniform_(self.positional_encoding.position_embedding, -.02, .02)
        
        # mask token
        trunc_normal_(self.mask_token, std=.02)


    def encoding(self, long_term_history, mask=True):
        """Encoding process of TSFormer: patchify, positional encoding, mask, Transformer layers.

        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, 1, P * L],
                                                which is used in the TSFormer.
                                                P is the number of segments (patches).
            mask (bool): True in pre-training stage and False in forecasting stage.

        Returns:
            torch.Tensor: hidden states of unmasked tokens
            list: unmasked token index
            list: masked token index
        """

        batch_size, num_nodes, _, _ = long_term_history.shape
        
        # Patchify and embed the multivariate timeseries.
        # NOTE: each patch is embedded into a higher dimensional space, e.g., from L(=12) to d(=96) in the METR-LA case.
        patches = self.patch_embedding(long_term_history)     # B, N, d, P
        patches = patches.transpose(-1, -2)                   # B, N, P, d
        print(f"DEBUG FRA, TSFormer.encoding => shape patches: {patches.shape}")
        

        ### positional embedding applied to the patches. ###
        patches = self.positional_encoding(patches)
        print(f"DEBUG FRA, TSFormer.encoding => shape patches after pos embedding: {patches.shape}")


        ### Patch masking ###
        if mask:
            # 1 - Create the masks to be applied on the patchified multivariate timeseries. The masks are simply indexes, and can be used to select 
            #     a subset of patches from the complete, patchified multivariate timeseries. When using the unmasked_token_index, this creates
            #     in output a smaller tensor, "encoder_input" that will be fed to the transformer layers.
            unmasked_token_index, masked_token_index = self.mask()
            # 2 - Select the tokens that haven't been masked.
            encoder_input = patches[:, :, unmasked_token_index, :]
        else:
            unmasked_token_index, masked_token_index = None, None
            encoder_input = patches
        print(f"DEBUG FRA, TSFormer.encoding => shape encoder_input after masking: {encoder_input.shape}")


        ### Transformer encoding ###
        hidden_states_unmasked = self.encoder(encoder_input)
        print(f"DEBUG FRA, TSFormer.encoding => shape hidden_states_unmasked: {hidden_states_unmasked.shape}")
        
        ### Final normalization ###
        # NOTE: the final view seems useless, it doesn't change the shape of hidden_states_unmasked.
        hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1, self.embed_dim)
        print(f"DEBUG FRA, TSFormer.encoding => shape hidden_states_unmasked after normalization and final reshaping: {hidden_states_unmasked.shape}")

        return hidden_states_unmasked, unmasked_token_index, masked_token_index


    def decoding(self, hidden_states_unmasked, masked_token_index):
        """Decoding process of TSFormer: encoder 2 decoder layer, add mask tokens, Transformer layers, predict.

        Args:
            hidden_states_unmasked (torch.Tensor): hidden states of unmasked tokens [B, N, P*(1-r), d]. NOTE: "r" is the percentage of masked tokens.
            masked_token_index (list): masked token index

        Returns:
            torch.Tensor: reconstructed data
        """

        batch_size, num_nodes, _, _ = hidden_states_unmasked.shape

        # encoder 2 decoder layer -- it's just a simple linear layer with bias term.
        # NOTE: it does not change the dimensionality but potentially adjusts the embeddings from the encoder for better decoding.
        hidden_states_unmasked = self.enc_2_dec_emb(hidden_states_unmasked)

        # Generate a tensor representing the masked tokens.
        # So, a masked token is essentially a zero value, i.e., nn.Parameter(torch.zeros(1, 1, 1, embed_dim)) in self.mask_token. 
        # First, we replicate a masked token via "expand" to create a tensor of zeros with shape [B, N, P*r, d].
        # Secondly, we add the positional embeddings on these masked tokens via self.positional_encoding by passing the masked_token_index, which enables
        # to take into account the position of the masked tokens when adding the encoding. 
        hidden_states_masked = self.positional_encoding(
            self.mask_token.expand(batch_size, num_nodes, len(masked_token_index), hidden_states_unmasked.shape[-1]),
            index=masked_token_index
            )
        
        print(f"DEBUG FRA, TSFormer.decoding => shape hidden_states_unmasked: {hidden_states_unmasked.shape}")
        print(f"DEBUG FRA, TSFormer.decoding => shape hidden_states_masked: {hidden_states_masked.shape}")
        # Here we concatenate the block of hidden_states pertaining the unmasked tokens, followed by the block of hidden_states
        # pertaining the masked tokens.
        # NOTE: the concatenation appears to destroy the temporal ordering across patches. Is this perhaps reconstructed outside of the decoding
        #       method?
        hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)   # B, N, P, d
        print(f"DEBUG FRA, TSFormer.decoding => shape hidden_states_full: {hidden_states_full.shape}")

        # Now the projected hidden states pertaining to unmasked tokens, as well as hidden states related to masked tokens,
        # are passed to Transformer layers, which are in charge of doing the appropriate transformations that will help to reconstruct
        # the patches in their original space (similar to the transformer layers stack of the encoder, just changes the number of layers).
        hidden_states_full = self.decoder(hidden_states_full)
        hidden_states_full = self.decoder_norm(hidden_states_full)
        print(f"DEBUG FRA, TSFormer.decoding => shape hidden_states_full / 2: {hidden_states_full.shape}")

        # prediction (reconstruction): here we have a simple linear layer, i.e., self.output_layer, that projects the 96-dim embeddings
        # of the patches back to the time series space (e.g., 12 samples with METR-LA).
        # NOTE: the view seems to do a useless reshape, at least in the METR-LA case.
        reconstruction_full = self.output_layer(hidden_states_full.view(batch_size, num_nodes, -1, self.embed_dim))
        print(f"DEBUG FRA, TSFormer.decoding => shape reconstruction_full / 2: {reconstruction_full.shape}")

        return reconstruction_full


    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, unmasked_token_index, masked_token_index):
        """Get reconstructed masked tokens and corresponding ground-truth for subsequent loss computing.

        Args:
            reconstruction_full (torch.Tensor): reconstructed full tokens.
            real_value_full (torch.Tensor): ground truth full tokens.
            unmasked_token_index (list): unmasked token index.
            masked_token_index (list): masked token index.

        Returns:
            torch.Tensor: reconstructed masked tokens.
            torch.Tensor: ground truth masked tokens.
        """
        # 1 - get reconstructed masked tokens
        batch_size, num_nodes, _, _ = reconstruction_full.shape
        
        # 1.1 - Retrieve the reconstructed masked tokens in reconstruction full -- it's a contiguous block due to previous cat. Shape [B, N, r*P, L],
        #       where L is the length of each patch in the original (timeseries) space.
        reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_index):, :]
        print(f"DEBUG FRA, TSFormer.get_reconstructed_masked_tokens => shape reconstruction_masked_tokens / 1: {reconstruction_masked_tokens.shape}")
        
        # 1.2 - Then, reshape (view) the tensor such that its shape becomes [B, r*P*L, N] -- we are essentially flattening the patch embeddings of each node.
        # Finally, swap (permute) the last two dimensions, yielding a tensor with shape [B, r*P*L, N]
        reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)
        print(f"DEBUG FRA, TSFormer.get_reconstructed_masked_tokens => shape reconstruction_masked_tokens / 2: {reconstruction_masked_tokens.shape}")


        # 2 - Retrieve the actual values of all the patches in a window in their original space.
        #     Permute the dimensions such that label_full has shape [B, L*P, N, num_features_ts]
        label_full = real_value_full.permute(0, 3, 1, 2)
        print(f"DEBUG FRA, TSFormer.get_reconstructed_masked_tokens => shape label_full / 1: {label_full.shape}")

        # 2.1 - Unfold accepts three parameters: dimension, size, and step. It returns a view of the original tensor which contains 
        # all slices of size "size" from self tensor in the dimension "dimension". So, what the unfold below is doing is to extract
        # windows of length "self.patch_size" by sliding a window of length "self.patch_size" over "label_full", and store them in a new dimension
        # that will be appended at the end of label_full. The new dimension will have size "self.patch_size", while dimension 1 (i.e., L*P) will
        # have a new size P. Overall, this yields a tensor with shape [B, P, N, num_features_ts, L]
        label_full = label_full.unfold(1, self.patch_size, self.patch_size)
        print(f"DEBUG FRA, TSFormer.get_reconstructed_masked_tokens => shape label_full / 2: {label_full.shape}")
        
        # 2.2 - Select only the feature of interest in the timeseries (in case of METR-LA we have only 1 feature per node's univariate timeseries).
        #       This eliminates the "num_features_ts" dimension. Finally, swap the first dimension with the second one, yielding a tensor with shape
        #       [B, N, P, L]. 
        label_full = label_full[:, :, :, self.selected_feature, :].transpose(1, 2)
        print(f"DEBUG FRA, TSFormer.get_reconstructed_masked_tokens => shape label_full / 3: {label_full.shape}")
        
        # 2.3 - Select the tokens that were masked, and ensure that they are arranged contiguously in the final tensor.
        label_masked_tokens = label_full[:, :, masked_token_index, :].contiguous() # B, N, r*P, L

        # 2.4 - Then reshape (view) yielding shape [B, N, r*P*L], and then transpose to get the same shape as "reconstruction_masked_tokens",
        #       i.e., [B, r*P*L, N], so that the reconstructed masked tokens can be compared with the ground truth.
        label_masked_tokens = label_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)

        return reconstruction_masked_tokens, label_masked_tokens


    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None,
                batch_seen: int = None, epoch: int = None, **kwargs) -> torch.Tensor:
        """feed forward of the TSFormer.
            TSFormer has two modes: the pre-training mode and the forecasting mode,
                                    which are used in the pre-training stage and the forecasting stage, respectively.

        Args:
            history_data (torch.Tensor): very long-term historical time series with shape B, L * P, N, 1.

        Returns:
            pre-training:
                torch.Tensor: the reconstruction of the masked tokens. Shape [B, L * P * r, N, 1]
                torch.Tensor: the ground truth of the masked tokens. Shape [B, L * P * r, N, 1]
                dict: data for plotting.
            forecasting:
                torch.Tensor: the output of TSFormer of the encoder with shape [B, N, L, 1].
        """

        # Reshape the array containing the currently considered batch of historical windows of multivariate timeseries.
        # Original history_data has shape (B, L*P, N, 1), i.e., Batch size, Length of the hist. time series expressed as L*P, # Nodes (which are the
        # ), # features (= 1)
        # NOTE 1: each element in the batch refers to a pair (hist window, future window), extracted from the original dataset in the preprocessing step
        #         by generating indexes using sliding windows.
        # NOTE 2: in the paper we have L=12. So, e.g. with METR-LA, P=2016/12=168. 
        history_data = history_data.permute(0, 2, 3, 1)     # B, N, 1, L * P


        # feed forward
        # Case 1 - pre-training mode.
        if self.mode == "pre-train":
            # 1.1 - encoding: applies patchification, positional encoding, masking, and then the actual transformer layers).
            hidden_states_unmasked, unmasked_token_index, masked_token_index = self.encoding(history_data)
            
            # 1.2 - decoding: here the decoder attempts to reconstruct the patches from the embeddings.
            #       NOTE: it appears that the decoder does not preserve the temporal order across patches when reconstructing it -- see
            #             the concatenation in "decoding".
            reconstruction_full = self.decoding(hidden_states_unmasked, masked_token_index)
            
            # Return the reconstructed tokens that were masked with their actual values from the dataset for subsequent loss computing
            reconstruction_masked_tokens, label_masked_tokens = \
                self.get_reconstructed_masked_tokens(reconstruction_full, history_data, unmasked_token_index, masked_token_index)
            return reconstruction_masked_tokens, label_masked_tokens
        
        # Case 2 - forecasting mode
        else:
            hidden_states_full, _, _ = self.encoding(history_data, mask=False)
            return hidden_states_full
