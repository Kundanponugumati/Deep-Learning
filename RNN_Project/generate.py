import torch

from data import tokenize
from model import RNNFromScratch


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    "checkpoints/rnn_model.pt",
    map_location="cpu",
    weights_only=False
)


# ============================================================
# RESTORE CONFIGURATION
# ============================================================

vocab = checkpoint["vocab"]
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

vocab_size = checkpoint["vocab_size"]
embedding_dim = checkpoint["embedding_dim"]
hidden_dim = checkpoint["hidden_dim"]

unk_id = stoi["<UNK>"]


# ============================================================
# RECREATE MODEL ARCHITECTURE
# ============================================================

model = RNNFromScratch(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim
)


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# GENERATION
# ============================================================

def generate(
    start_text,
    num_tokens=50,
    temperature=0.8
):

    tokens = tokenize(start_text)

    token_ids = [
        stoi.get(token, unk_id)
        for token in tokens
    ]

    h = torch.zeros(
        1,
        hidden_dim
    )

    generated_tokens = tokens.copy()

    with torch.no_grad():

        # ----------------------------------------------------
        # Process prompt
        # ----------------------------------------------------

        for token_id in token_ids:

            token_tensor = torch.tensor(
                [token_id],
                dtype=torch.long
            )

            logits, h = model.forward_step(
                token_tensor,
                h
            )

        # ----------------------------------------------------
        # Generate new tokens
        # ----------------------------------------------------

        for _ in range(num_tokens):

            probabilities = torch.softmax(
                logits / temperature,
                dim=-1
            )

            predicted_id = torch.multinomial(
                probabilities,
                num_samples=1
            ).squeeze(1)

            predicted_token = itos[
                predicted_id.item()
            ]

            generated_tokens.append(
                predicted_token
            )

            # Predicted token becomes next input
            logits, h = model.forward_step(
                predicted_id,
                h
            )

    return " ".join(generated_tokens)


# ============================================================
# TEST
# ============================================================

print(
    generate(
        start_text="hello, good morn ",
        num_tokens=5,
        temperature=0.8
    )
)