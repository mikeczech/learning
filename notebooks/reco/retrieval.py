from typing import List
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

class QueryTower(nn.Module):

    def __init__(self, user_ids: List[str], user_emb_dim: int= 16):
        super(QueryTower, self).__init__()

        self.user_id_to_index = {user_id: i for i, user_id in enumerate(user_ids)}
        self.user_emb = nn.Embedding(len(user_ids), user_emb_dim)
        self.normalized_age = nn.BatchNorm1d(1)
        self.linear = nn.Linear(user_emb_dim + 1, user_emb_dim)

    def forward(self, customer_ids: List["str"], ages: torch.Tensor):
        user_indices = self.get_user_indices(customer_ids)
        user_features = self.user_emb(user_indices)

        age_features = self.normalized_age(ages.reshape(-1, 1))

        features = torch.cat([user_features, age_features], dim=1)

        return self.linear(features)

    def get_user_indices(self, user_ids):
        return torch.tensor([self.user_id_to_index[user_id] for user_id in user_ids], dtype=torch.long)


class ItemTower(nn.Module):

    def __init__(self):
        pass

    def forward(self, x):
        pass
