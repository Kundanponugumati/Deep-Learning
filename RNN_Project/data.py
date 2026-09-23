import re
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):
    text = text.lower()
    return re.findall(r"\w+|[^\w\s]", text)


# ============================================================
# DATASET
# ============================================================

class RNNDataset(Dataset):

    def __init__(self, token_ids, sequence_length):
        self.token_ids = token_ids
        self.sequence_length = sequence_length

    def __len__(self):
        return (len(self.token_ids) - 1) // self.sequence_length

    def __getitem__(self, index):

        start = index * self.sequence_length
        end = start + self.sequence_length

        x = self.token_ids[start:end]
        y = self.token_ids[start + 1:end + 1]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    max_vocab_size=5000,
    sequence_length=30,
    batch_size=32
):

    # Load WikiText
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1"
    )

    train_data = dataset["train"]

    # --------------------------------------------------------
    # Tokenize entire training corpus
    # --------------------------------------------------------

    all_tokens = []

    for row in train_data:
        tokens = tokenize(row["text"])
        all_tokens.extend(tokens)

    # --------------------------------------------------------
    # Build vocabulary
    # --------------------------------------------------------

    token_counts = Counter(all_tokens)

    most_common_tokens = token_counts.most_common(
        max_vocab_size - 2
    )

    vocab_tokens = [
        token
        for token, count in most_common_tokens
    ]

    vocab = [
        "<PAD>",
        "<UNK>"
    ] + vocab_tokens

    stoi = {
        token: index
        for index, token in enumerate(vocab)
    }

    itos = {
        index: token
        for token, index in stoi.items()
    }

    # --------------------------------------------------------
    # Convert corpus → IDs
    # --------------------------------------------------------

    unk_id = stoi["<UNK>"]

    all_token_ids = [
        stoi.get(token, unk_id)
        for token in all_tokens
    ]

    # --------------------------------------------------------
    # Dataset + DataLoader
    # --------------------------------------------------------

    train_dataset = RNNDataset(
        all_token_ids,
        sequence_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader, vocab, stoi, itos