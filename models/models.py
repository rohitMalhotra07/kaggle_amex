import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, feat_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = nn.MultiheadAttention(feat_dim, num_heads, batch_first = True)
        self.ffn = nn.Sequential(
            nn.Sequential(
                nn.Linear(feat_dim,ff_dim),
                nn.GELU()
            ),
             nn.Linear(ff_dim, feat_dim)
        )
        
        self.layernorm1 = nn.LayerNorm(feat_dim, eps = 1e-6)
        self.layernorm2 = nn.LayerNorm(feat_dim, eps = 1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
    
    def forward(self, inputs):
        attn_output, attn_output_weights = self.att(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output)
        
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2
    

    
class AttentionModel(nn.Module):
    def __init__(self, embed_dim, feat_dim, num_heads, ff_dim, rate=0.1, num_blocks=2):
        super().__init__()
        
        self.embedding_layers = nn.ModuleList()
        for k in range(11):
            emb = nn.Embedding(10,4)
            self.embedding_layers.append(emb)
        
        self.dense_layers1 = nn.Linear(feat_dim-11+(4*11),feat_dim)
        
        self.transformer_layers = nn.ModuleList()
        for k in range(num_blocks):
            transformer_block = TransformerBlock(feat_dim, num_heads, ff_dim, rate)
            self.transformer_layers.append(transformer_block)
        
        
        # CLASSIFICATION HEAD
        self.dense_layers2 = nn.Sequential(nn.Linear(feat_dim, 64),nn.ReLU())
        self.dense_layers3 = nn.Sequential(nn.Linear(64, 32),nn.ReLU())
        self.output_layer = nn.Sequential(nn.Linear(32, 1),nn.Sigmoid())
        
    
    def forward(self, inp):
        embeddings = [inp[:,:,11:]]
        for k,emb in enumerate(self.embedding_layers):
            tmp = emb(inp[:,:,k].int())
            embeddings.append(tmp)
        x = torch.cat(embeddings, dim=2)
        x = self.dense_layers1(x)
        
        # TRANSFORMER BLOCKS
        for k , transformer_block in enumerate(self.transformer_layers):
            x_old = x
            x = transformer_block(x)
            x = 0.9*x + 0.1*x_old # SKIP CONNECTION
        
        # CLASSIFICATION HEAD
        x = self.dense_layers2(x[:,-1,:])
        x = self.dense_layers3(x)
        outputs = torch.squeeze(self.output_layer(x))
        
        return outputs
        