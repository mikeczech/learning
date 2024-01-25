from typing import List
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryTower(nn.Module):
    def __init__(
        self, user_ids: List[str], user_emb_dim: int = 16, output_dim: int = 10
    ):
        super(QueryTower, self).__init__()

        self.user_id_to_index = {user_id: i for i, user_id in enumerate(user_ids)}
        self.user_embedding = nn.Embedding(len(user_ids), user_emb_dim)
        self.normalized_age = nn.BatchNorm1d(1)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(
            user_emb_dim + 1, output_dim
        )  # what woulde be a good target dimension?

    def forward(self, customer_ids: List[str], ages: torch.Tensor):
        user_indices = self.get_user_indices(customer_ids)
        user_features = self.user_embedding(user_indices)

        age_features = self.normalized_age(ages.reshape(-1, 1))

        features = torch.cat([user_features, age_features], dim=1)
        features = self.relu(features)

        return self.linear(features)

    def get_user_indices(self, user_ids: List[str]):
        return torch.tensor(
            [self.user_id_to_index[user_id] for user_id in user_ids], dtype=torch.long
        )


class ItemTower(nn.Module):
    def __init__(
        self,
        item_ids: List[str],
        index_group_names: List[str],
        garment_group_names: List[str],
        item_emb_dim: int = 16,
        output_dim: int = 10,
    ):
        super(ItemTower, self).__init__()

        self.item_to_id_index = {item_id: i for i, item_id in enumerate(item_ids)}
        self.index_group_to_index = {
            name: i for i, name in enumerate(index_group_names)
        }
        self.garment_group_to_index = {
            name: i for i, name in enumerate(garment_group_names)
        }
        self.item_embedding = nn.Embedding(len(item_ids), item_emb_dim)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(
            item_emb_dim + len(index_group_names) + len(garment_group_names),
            output_dim,
        )

    def forward(
        self,
        item_ids: List[str],
        index_group_names: List[str],
        garment_group_names: List[str],
    ):
        item_indices = self.get_item_indices(item_ids)
        item_features = self.item_embedding(item_indices)

        index_group_indices = self.get_index_group_indices(index_group_names)
        garment_group_indices = self.get_garment_group_index(garment_group_names)

        index_group_features = F.one_hot(
            index_group_indices, num_classes=len(self.index_group_to_index)
        )
        garment_group_features = F.one_hot(
            garment_group_indices, num_classes=len(self.garment_group_to_index)
        )

        features = torch.cat(
            [item_features, index_group_features, garment_group_features], dim=1
        )
        features = self.relu(features)

        return self.linear(features)

    def get_item_indices(self, item_ids: List[str]):
        return torch.tensor(
            [self.item_to_id_index[item_id] for item_id in item_ids], dtype=torch.long
        )

    def get_index_group_indices(self, index_group_names: List[str]):
        return torch.tensor(
            [self.index_group_to_index[name] for name in index_group_names],
            dtype=torch.long,
        )

    def get_garment_group_index(self, garment_group_names: List[str]):
        return torch.tensor(
            [self.garment_group_to_index[name] for name in garment_group_names],
            dtype=torch.long,
        )
