import re
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


# ============================================================
# 1. LOAD DATASET
# ============================================================

dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1"
)

train_data = dataset["train"]

print("Number of training rows:", len(train_data))


# ============================================================
# 2. TOKENIZER
# ============================================================

def tokenize(text):

    # Normalize case
    text = text.lower()

    # Separate words and punctuation
    tokens = re.findall(r"\w+|[^\w\s]", text)

    return tokens


# ============================================================
# 3. TOKENIZE ENTIRE TRAINING DATA
# ============================================================

all_tokens = []

for row in train_data:

    text = row["text"]

    tokens = tokenize(text)

    all_tokens.extend(tokens)


print("Total number of tokens:", len(all_tokens))
print("First 30 tokens:")
print(all_tokens[:30])


# ============================================================
# 4. BUILD VOCABULARY
# ============================================================

token_counts = Counter(all_tokens)

max_vocab_size = 5000

# Reserve:
# 0 -> <PAD>
# 1 -> <UNK>

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


# String → Integer
stoi = {
    token: index
    for index, token in enumerate(vocab)
}


# Integer → String
itos = {
    index: token
    for token, index in stoi.items()
}


vocab_size = len(vocab)

print("\nVocabulary size:", vocab_size)

print("\nFirst few vocabulary entries:")

for i in range(20):
    print(i, itos[i])


# ============================================================
# 5. CONVERT ENTIRE TRAINING CORPUS TO TOKEN IDs
# ============================================================

unk_id = stoi["<UNK>"]

all_token_ids = [
    stoi.get(token, unk_id)
    for token in all_tokens
]


print("\nFirst 30 token IDs:")
print(all_token_ids[:30])

print("\nDecoded again:")
print([
    itos[token_id]
    for token_id in all_token_ids[:30]
])


# ============================================================
# 6. CREATE DATASET
# ============================================================

class RNNDataset(Dataset):

    def __init__(
        self,
        token_ids,
        sequence_length
    ):

        self.token_ids = token_ids
        self.sequence_length = sequence_length


    def __len__(self):

        # Non-overlapping chunks
        return (
            len(self.token_ids) - 1
        ) // self.sequence_length


    def __getitem__(self, index):

        start = index * self.sequence_length

        end = start + self.sequence_length


        # Input sequence
        x = self.token_ids[
            start:end
        ]


        # Same sequence shifted by 1
        y = self.token_ids[
            start + 1:end + 1
        ]


        return (
            torch.tensor(
                x,
                dtype=torch.long
            ),
            torch.tensor(
                y,
                dtype=torch.long
            )
        )


# ============================================================
# 7. CREATE DATALOADER
# ============================================================

sequence_length = 30
batch_size = 32


train_dataset = RNNDataset(
    all_token_ids,
    sequence_length
)


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)


print(
    "\nNumber of training sequences:",
    len(train_dataset)
)


# ============================================================
# 8. INSPECT ONE BATCH
# ============================================================

X, Y = next(iter(train_loader))


print("\nX shape:")
print(X.shape)

print("\nY shape:")
print(Y.shape)


print("\nFirst input sequence:")

print([
    itos[token.item()]
    for token in X[0]
])


print("\nFirst target sequence:")

print([
    itos[token.item()]
    for token in Y[0]
])


# ============================================================
# 9. BUILD RNN FROM SCRATCH
# ============================================================

class RNNFromScratch(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim
    ):

        super().__init__()

        self.hidden_dim = hidden_dim


        # ----------------------------------------------------
        # TOKEN EMBEDDING
        # ----------------------------------------------------

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )


        # ----------------------------------------------------
        # INPUT -> HIDDEN
        #
        # [embedding_dim, hidden_dim]
        # ----------------------------------------------------

        self.Wxh = nn.Parameter(
            torch.randn(
                embedding_dim,
                hidden_dim
            ) * 0.01
        )


        # ----------------------------------------------------
        # HIDDEN -> HIDDEN
        #
        # [hidden_dim, hidden_dim]
        # ----------------------------------------------------

        self.Whh = nn.Parameter(
            torch.randn(
                hidden_dim,
                hidden_dim
            ) * 0.01
        )


        # Hidden bias
        self.bh = nn.Parameter(
            torch.zeros(hidden_dim)
        )


        # ----------------------------------------------------
        # HIDDEN -> VOCABULARY
        #
        # [hidden_dim, vocab_size]
        # ----------------------------------------------------

        self.Why = nn.Parameter(
            torch.randn(
                hidden_dim,
                vocab_size
            ) * 0.01
        )


        # Vocabulary/output bias
        self.by = nn.Parameter(
            torch.zeros(vocab_size)
        )


    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(self, x):

        # x:
        #
        # [batch_size, sequence_length]

        batch_size, sequence_length = x.shape


        # ----------------------------------------------------
        # EMBEDDING
        # ----------------------------------------------------

        embedded = self.embedding(x)

        # embedded:
        #
        # [batch_size,
        #  sequence_length,
        #  embedding_dim]


        # ----------------------------------------------------
        # INITIAL HIDDEN STATE
        # ----------------------------------------------------

        h = torch.zeros(
            batch_size,
            self.hidden_dim,
            device=x.device
        )

        # h:
        #
        # [batch_size, hidden_dim]


        outputs = []


        # ----------------------------------------------------
        # PROCESS ONE TIMESTEP AT A TIME
        # ----------------------------------------------------

        for t in range(sequence_length):


            # Get every sequence's token
            # at timestep t

            x_t = embedded[:, t, :]

            # x_t:
            #
            # [batch_size, embedding_dim]


            # ------------------------------------------------
            # RNN EQUATION
            # ------------------------------------------------

            h = torch.tanh(

                x_t @ self.Wxh

                +

                h @ self.Whh

                +

                self.bh
            )

            # h:
            #
            # [batch_size, hidden_dim]


            # ------------------------------------------------
            # PREDICT NEXT TOKEN
            # ------------------------------------------------

            logits = (

                h @ self.Why

                +

                self.by
            )

            # logits:
            #
            # [batch_size, vocab_size]


            outputs.append(logits)


        # ----------------------------------------------------
        # COMBINE ALL TIMESTEPS
        # ----------------------------------------------------

        outputs = torch.stack(
            outputs,
            dim=1
        )

        # outputs:
        #
        # [batch_size,
        #  sequence_length,
        #  vocab_size]


        return outputs

    def forward_step(self, token, h):

        embedded = self.embedding(token)

        h = torch.tanh(
            embedded @ self.Wxh
            +
            h @ self.Whh
            +
            self.bh
        )

        logits = (
            h @ self.Why
            +
            self.by
        )

        return logits, h

# ============================================================
# 10. CREATE MODEL
# ============================================================

embedding_dim = 128
hidden_dim = 256


model = RNNFromScratch(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim
)


print("\nModel:")
print(model)


# ============================================================
# 11. TEST FORWARD PASS
# ============================================================

X, Y = next(iter(train_loader))


logits = model(X)


print("\nInput shape:")
print(X.shape)

print("\nTarget shape:")
print(Y.shape)

print("\nLogits shape:")
print(logits.shape)


# ============================================================
# 12. LOSS FUNCTION
# ============================================================

loss_fn = nn.CrossEntropyLoss()


# Flatten:
#
# [batch, sequence, vocab]
#
# ->
#
# [batch * sequence, vocab]

logits_flat = logits.reshape(
    -1,
    vocab_size
)


# [batch, sequence]
#
# ->
#
# [batch * sequence]

targets_flat = Y.reshape(-1)


print("\nFlattened logits:")
print(logits_flat.shape)

print("\nFlattened targets:")
print(targets_flat.shape)


initial_loss = loss_fn(
    logits_flat,
    targets_flat
)


print(
    "\nInitial loss:",
    initial_loss.item()
)


# ============================================================
# 13. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 14. TRAINING LOOP
# ============================================================

num_epochs = 3


for epoch in range(num_epochs):

    model.train()

    total_loss = 0


    for batch_number, (X, Y) in enumerate(train_loader):


        # ----------------------------------------------------
        # 1. CLEAR OLD GRADIENTS
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # 2. FORWARD PASS
        # ----------------------------------------------------

        logits = model(X)


        # ----------------------------------------------------
        # 3. CALCULATE LOSS
        # ----------------------------------------------------

        loss = loss_fn(

            logits.reshape(
                -1,
                vocab_size
            ),

            Y.reshape(-1)
        )


        # ----------------------------------------------------
        # 4. BACKPROPAGATION THROUGH TIME
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # 5. GRADIENT CLIPPING
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        # ----------------------------------------------------
        # 6. UPDATE PARAMETERS
        # ----------------------------------------------------

        optimizer.step()


        total_loss += loss.item()


        # Print occasionally
        if batch_number % 100 == 0:

            print(
                f"Epoch {epoch + 1} "
                f"Batch {batch_number} "
                f"Loss {loss.item():.4f}"
            )


    average_loss = (
        total_loss /
        len(train_loader)
    )


    print(
        f"\nEpoch {epoch + 1}/{num_epochs} "
        f"Average Loss: {average_loss:.4f}\n"
    )


print("Training finished.")


def forward_step(self, token, h):

    embedded = self.embedding(token)

    h = torch.tanh(
        embedded @ self.Wxh
        +
        h @ self.Whh
        +
        self.bh
    )

    logits = (
        h @ self.Why
        +
        self.by
    )

    return logits, h

def generate(
    model,
    start_text,
    num_tokens=20
):

    model.eval()

    tokens = tokenize(start_text)

    token_ids = [
        stoi.get(token, unk_id)
        for token in tokens
    ]

    h = torch.zeros(
        1,
        model.hidden_dim
    )

    generated_tokens = tokens.copy()

    with torch.no_grad():

        # ------------------------------------
        # Process prompt
        # ------------------------------------

        for token_id in token_ids:

            token_tensor = torch.tensor(
                [token_id],
                dtype=torch.long
            )

            logits, h = model.forward_step(
                token_tensor,
                h
            )


        # ------------------------------------
        # Generate new tokens
        # ------------------------------------

        for _ in range(num_tokens):

            predicted_id = torch.argmax(
                logits,
                dim=-1
            )

            predicted_token = itos[
                predicted_id.item()
            ]

            generated_tokens.append(
                predicted_token
            )

            # Feed prediction back into RNN

            logits, h = model.forward_step(
                predicted_id,
                h
            )


    return " ".join(generated_tokens)


text = generate(
    model,
    start_text="the",
    num_tokens=30
)

print(text)