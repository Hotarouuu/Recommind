import torch
import torch.nn as nn

class NeuMF(nn.Module):
    def __init__(self, n_users, n_items,n_genders, n_authors, n_factors=8):
        super().__init__()

        # Embeddings for GMF (Generalized Matrix Factorization) path
        self.user_gmf = nn.Embedding(n_users, n_factors)
        self.item_gmf = nn.Embedding(n_items, n_factors)

        # Embeddings for MLP path
        self.user_mlp = nn.Embedding(n_users, n_factors)
        self.item_mlp = nn.Embedding(n_items, n_factors)
        self.item_gender_emb = nn.Embedding(n_genders, n_factors)
        self.item_authors_emb = nn.Embedding(n_authors, n_factors)

        
        # Initialize embeddings with small uniform values
        self.user_gmf.weight.data.uniform_(0, 0.05)
        self.item_gmf.weight.data.uniform_(0, 0.05)
        self.user_mlp.weight.data.uniform_(0, 0.05)
        self.item_mlp.weight.data.uniform_(0, 0.05)

        # MLP input: user + item embedding + text embedding
        input_dim = n_factors * 2

        # MLP: several layers with ReLU and Dropout to prevent overfitting
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 32)
        )

        # Final prediction layer: GMF + MLP outputs concatenated
        self.final_layer = nn.Linear(n_factors + 32, 1)

    def forward(self, data):
        users = data[:, 1]
        items = data[:, 0]
        genders = data[:, 3]
        authors = data[:, 2]
        ratingsCount = data[:, 4]



        # GMF path
        gmf_user = self.user_gmf(users)
        gmf_item = self.item_gmf(items)
        gmf_out = gmf_user * gmf_item  # element-wise product

        # Gender and authors 
        gender_emb = self.item_gender_emb(genders)
        authors_emb = self.item_authors_emb(authors)


        # MLP path
        mlp_user = self.user_mlp(users)
        mlp_item = self.item_mlp(items)
        mlp_items = torch.cat([mlp_item, gender_emb, authors_emb,ratingsCount.unsqueeze(1)], dim=1)

        # Concatenate user, item, and text embeddings
        mlp_input = torch.cat([mlp_user, mlp_item], dim=1)
       # print(mlp_input.shape)
        mlp_out = self.mlp(mlp_input)

        # Combine GMF and MLP paths and make final prediction
        final_input = torch.cat([gmf_out, mlp_out], dim=1)
        out = self.final_layer(final_input).squeeze(1)

        return out