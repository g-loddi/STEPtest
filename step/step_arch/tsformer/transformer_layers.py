import math
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerLayers(nn.Module):
    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
        '''
        Args:

            hidden_dim: Dimension of the input embeddings (and the model). This is the dimension of each patch once brought in a higher dimensional
                        space by the application of a Conv1D block, e.g., from 12 to 96.
            mlp_ratio: Multiplier for the hidden dimension in the feedforward network inside each Transformer layer. Set to e.g. 4 in the METR-LA case.
        '''

        super().__init__()
        
        self.d_model = hidden_dim
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout) # Configure the TransformerEncoderLayers.
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers) # Stack "nlayers" of "encoder_layers".

        print(f"DEBUG FRA, TransformerLayers.__init__() => self.transformer_encoder = {self.transformer_encoder}")


    def forward(self, src):
        B, N, L, D = src.shape
        
        # Scale the input by the square root of the model dimension. common practice in Transformer models to stabilize
        # the learning process, particularly the initial stages of training.
        src = src * math.sqrt(self.d_model)
        
        # Reshape so that we have a 3D tensor, which is needed by the transformer layers.
        src = src.view(B*N, L, D) # Shape [B*N, L, D]
        
        # Swap the first two dimensions: from [B*N, L, D] to [L, B*N, D]. Required because the TransformerEncoder 
        # expects the input to have the shape [sequence_length, batch_size, embedding_dim], which in our case is
        # [# patches, number of univariate time series in a batch, dim embedding patches].
        src = src.transpose(0, 1)
        
        # Give the unmasked tokens as input to the encoder layers.
        # NOTE: Output contains the final hidden_states of the transformer layers, which includes the updated embeddings for each position in the sequence, 
        # incorporating information from all other positions due to the self-attention mechanism.
        output = self.transformer_encoder(src, mask=None)
        
        # Undo the previous transposition and reshaping.
        output = output.transpose(0, 1).view(B, N, L, D)
        
        return output
