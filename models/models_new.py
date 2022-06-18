import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads,ff_dim, rate=0.1):
        super().__init__()
        
        self.att = nn.MultiheadAttention(model_dim, num_heads, batch_first = True)
        self.ffn = nn.Sequential(
            nn.Sequential(
                nn.Linear(model_dim,ff_dim),
                nn.GELU()
            ),
             nn.Linear(ff_dim, model_dim)
        )
        
        self.layernorm1 = nn.LayerNorm(model_dim, eps = 1e-6)
        self.layernorm2 = nn.LayerNorm(model_dim, eps = 1e-6)
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
        
        self.dense_layers1 = nn.Sequential(nn.Linear(feat_dim-11+(4*11),feat_dim),nn.ReLU())
        self.dense_layers1a = nn.Sequential(nn.Linear(feat_dim,embed_dim),nn.ReLU())
        
        self.transformer_layers = nn.ModuleList()
        for k in range(num_blocks):
            transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
            self.transformer_layers.append(transformer_block)
        
        
        # CLASSIFICATION HEAD
        self.dense_layers2 = nn.Sequential(nn.Linear(embed_dim, 64),nn.ReLU())
        self.dense_layers3 = nn.Sequential(nn.Linear(64, 32),nn.ReLU())
        self.output_layer = nn.Sequential(nn.Linear(32, 1),nn.Sigmoid())
        
    
    def forward(self, inp):
        embeddings = [inp[:,:,11:]]
        for k,emb in enumerate(self.embedding_layers):
            tmp = emb(inp[:,:,k].int())
            embeddings.append(tmp)
        x = torch.cat(embeddings, dim=2)
        x = self.dense_layers1(x)
        x = self.dense_layers1a(x)
        
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
    

class GRUModel(nn.Module):
    def __init__(self, embed_dim, feat_dim , rate=0.1, num_layers=2):
        super().__init__()
        
        self.embedding_layers = nn.ModuleList()
        for k in range(11):
            emb = nn.Embedding(10,4)
            self.embedding_layers.append(emb)
        
        self.dense_layers1 = nn.Sequential(nn.Linear(feat_dim-11+(4*11),feat_dim),nn.ReLU())
        self.dense_layers1a = nn.Sequential(nn.Linear(feat_dim,embed_dim),nn.ReLU())
        
        self.GRUBlock = nn.GRU(input_size = embed_dim, hidden_size = embed_dim, num_layers = num_layers, batch_first = True, dropout = rate)
        
        
        # CLASSIFICATION HEAD
        self.dense_layers2 = nn.Sequential(nn.Linear(embed_dim, 64),nn.ReLU())
        self.dense_layers3 = nn.Sequential(nn.Linear(64, 32),nn.ReLU())
        self.output_layer = nn.Sequential(nn.Linear(32, 1),nn.Sigmoid())
        
    
    def forward(self, inp):
        embeddings = [inp[:,:,11:]]
        for k,emb in enumerate(self.embedding_layers):
            tmp = emb(inp[:,:,k].int())
            embeddings.append(tmp)
        x = torch.cat(embeddings, dim=2)
        x = self.dense_layers1(x)
        x = self.dense_layers1a(x)
        
        x,_ = self.GRUBlock(x)
        
        # CLASSIFICATION HEAD
        x = self.dense_layers2(x[:,-1,:])
        x = self.dense_layers3(x)
        outputs = torch.squeeze(self.output_layer(x))
        
        return outputs
    


class GRUModel2(nn.Module):
    def __init__(self, embed_dim, feat_dim , rate=0.1, num_layers=2):
        super().__init__()
        
        self.embedding_layers = nn.ModuleList()
        for k in range(11):
            emb = nn.Embedding(10,4)
            self.embedding_layers.append(emb)
        
        self.dense_layers1 = nn.Sequential(nn.Linear(feat_dim-11+(4*11),feat_dim),nn.ReLU())
        self.dense_layers1a = nn.Sequential(nn.Linear(feat_dim,embed_dim),nn.ReLU())
        
        self.GRUBlock = nn.GRU(input_size = embed_dim, hidden_size = embed_dim, num_layers = num_layers, batch_first = True, dropout = rate)
        
        
        # CLASSIFICATION HEAD
        self.dense_layers2 = nn.Sequential(nn.Linear(embed_dim, 64),nn.ReLU())
        self.dense_layers3 = nn.Sequential(nn.Linear(64, 32),nn.ReLU())
        self.dense_layers4 = nn.Sequential(nn.Linear(32, 16),nn.ReLU())
        self.dense_layers5 = nn.Sequential(nn.Linear(16, 8),nn.ReLU())
        
        self.dense_layers6 = nn.Sequential(nn.Linear(8*13, 32),nn.ReLU())
        self.output_layer = nn.Sequential(nn.Linear(32, 1),nn.Sigmoid())
        
    
    def forward(self, inp):
        embeddings = [inp[:,:,11:]]
        for k,emb in enumerate(self.embedding_layers):
            tmp = emb(inp[:,:,k].int())
            embeddings.append(tmp)
        x = torch.cat(embeddings, dim=2)
        x = self.dense_layers1(x)
        x = self.dense_layers1a(x)
        
        x,_ = self.GRUBlock(x)
        
        # CLASSIFICATION HEAD
        x = self.dense_layers2(x)
        x = self.dense_layers3(x)
        x = self.dense_layers4(x)
        x = self.dense_layers5(x).flatten(start_dim=1)
        
        x = self.dense_layers6(x)
        outputs = torch.squeeze(self.output_layer(x))
        
        return outputs

    


    
class AttentionModelConv1d(nn.Module):
    def __init__(self, embed_dim, feat_dim, num_heads, ff_dim, rate=0.1, num_blocks=2, seq_length=13):
        super().__init__()
        
        self.embedding_layers = nn.ModuleList()
        for k in range(11):
            emb = nn.Embedding(10,4)
            self.embedding_layers.append(emb)
        
        self.dense_layers1 = nn.Sequential(nn.Linear(feat_dim-11+(4*11),feat_dim),nn.ReLU())
        # self.dense_layers1a = nn.Sequential(nn.Linear(feat_dim,embed_dim),nn.ReLU())
        
        self.conv1d = nn.Conv1d(seq_length, 7, kernel_size=3, stride=2, padding=3)
        
        model_dim = 96
        self.transformer_layers = nn.ModuleList()
        for k in range(num_blocks):
            transformer_block = TransformerBlock(model_dim, num_heads, ff_dim, rate)
            self.transformer_layers.append(transformer_block)
        
        
        self.conv1d_2 = nn.Conv1d(7, 1, kernel_size=1)
        # CLASSIFICATION HEAD
        self.dense_layers2 = nn.Sequential(nn.Linear(model_dim, 64),nn.ReLU())
        self.dense_layers3 = nn.Sequential(nn.Linear(64, 32),nn.ReLU())
        self.output_layer = nn.Sequential(nn.Linear(32, 1),nn.Sigmoid())
        
    
    def forward(self, inp):
        embeddings = [inp[:,:,11:]]
        for k,emb in enumerate(self.embedding_layers):
            tmp = emb(inp[:,:,k].int())
            embeddings.append(tmp)
        x = torch.cat(embeddings, dim=2)
        x = self.dense_layers1(x)
        # print(x.shape)
        x = self.conv1d(x)
        # print(x.shape)
        
        # TRANSFORMER BLOCKS
        for k , transformer_block in enumerate(self.transformer_layers):
            x_old = x
            x = transformer_block(x)
            x = 0.9*x + 0.1*x_old # SKIP CONNECTION
        
        x = self.conv1d_2(x).flatten(start_dim=1)
        
        # CLASSIFICATION HEAD
        x = self.dense_layers2(x)
        x = self.dense_layers3(x)
        outputs = torch.squeeze(self.output_layer(x))
        
        return outputs