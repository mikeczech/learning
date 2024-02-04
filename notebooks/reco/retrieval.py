from typing import List
from collections import Counter

from polars import DataFrame

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import Tensor


class InteractionDataset(Dataset):
    def __init__(self, df: DataFrame, device):
        self.df = df
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)
        customer_id = row["encoded_customer_id"]
        article_id = row["encoded_article_id"]
        age = torch.tensor(row["age"], dtype=torch.float).to(self.device)
        index_group_name = row["encoded_index_group_name"]
        garment_group_name = row["encoded_garment_group_name"]

        return customer_id, article_id, age, index_group_name, garment_group_name


class QueryTower(nn.Module):
    def __init__(
        self, num_users: int, device, user_emb_dim: int = 16, output_dim: int = 10
    ):
        super(QueryTower, self).__init__()

        self.user_embedding = nn.Embedding(num_users, user_emb_dim)
        self.normalized_age = nn.BatchNorm1d(1)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(
            user_emb_dim + 1, output_dim
        )  # what woulde be a good target dimension?
        self.device = device

    def forward(self, user_ids: Tensor, ages: Tensor):
        user_features = self.user_embedding(user_ids)
        age_features = self.normalized_age(ages.reshape(-1, 1))

        features = torch.cat([user_features, age_features], dim=1)
        features = self.relu(features)

        return self.linear(features)


class ItemTower(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_index_group_names: int,
        num_garment_group_names: int,
        device,
        item_emb_dim: int = 16,
        output_dim: int = 10,
    ):
        super(ItemTower, self).__init__()

        self.num_index_group_names = num_index_group_names
        self.num_garment_group_names = num_garment_group_names

        self.item_embedding = nn.Embedding(num_items, item_emb_dim)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(
            item_emb_dim + num_index_group_names + num_garment_group_names,
            output_dim,
        )
        self.device = device

    def forward(
        self,
        item_ids: Tensor,
        index_group_names: Tensor,
        garment_group_names: Tensor,
    ):
        item_features = self.item_embedding(item_ids).to(self.device)

        index_group_features = F.one_hot(
            index_group_names, num_classes=self.num_index_group_names
        )

        garment_group_features = F.one_hot(
            garment_group_names, num_classes=self.num_garment_group_names
        )

        features = torch.cat(
            [item_features, index_group_features, garment_group_features], dim=1
        )
        features = self.relu(features)

        return self.linear(features)


class TwoTowerModel(nn.Module):
    def __init__(self, query_model, item_model, device):
        super(TwoTowerModel, self).__init__()
        self._query_model = query_model
        self._item_model = item_model
        self.device = device

    def forward(
        self,
        customer_ids: Tensor,
        item_ids: Tensor,
        ages: Tensor,
        index_group_names: Tensor,
        garment_group_names: Tensor,
    ):
        query_embedding = self._query_model(customer_ids, ages)
        item_embedding = self._item_model(
            item_ids, index_group_names, garment_group_names
        )

        scores = torch.matmul(query_embedding, item_embedding.t())
        scores = F.log_softmax(scores, dim=1)  # NLLoss requires log probabilites

        num_queries = query_embedding.shape[0]
        num_items = item_embedding.shape[0]

        labels = torch.argmax(torch.eye(num_queries, num_items).to(self.device), dim=1)

        loss = F.nll_loss(scores, labels)

        return loss
