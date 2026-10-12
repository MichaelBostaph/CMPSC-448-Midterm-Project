"""Independently evaluate CNN and BiGRU classifiers on Chatbot Arena conversation_a.

Select which parts of conversation_a are used as model input:
  --input-mode user       Use only user messages.
  --input-mode assistant  Use only assistant/LLM messages (default).
  --input-mode both       Use user and assistant messages, preserving turn order.

The target is model_a. CNN and BiGRU train and evaluate independently, but print
identical metric fields:
CNN epoch N: loss=... accuracy=... macro_f1=...
RNN epoch N: loss=... accuracy=... macro_f1=...

After each epoch, the script also prints preview input text, the expected model_a
label, and each classifier's independent predicted output.

Before running, accept the Hugging Face dataset conditions, then run:
    huggingface-cli login

Examples:
    python Midterm_Project.py --rows 5000 --epochs 5 --input-mode user
    python Midterm_Project.py --rows 5000 --epochs 5 --input-mode assistant
    python Midterm_Project.py --rows 5000 --epochs 5 --input-mode both
    python Midterm_Project.py --show-examples 2 --preview-chars 500
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

DATASET_NAME = "lmsys/chatbot_arena_conversations"
TOKENIZER_NAME = "distilbert-base-uncased"
USER_ROLES = {"user", "human"}
ASSISTANT_ROLES = {"assistant", "bot", "model"}


def set_seed(seed):
    """Set random seeds and deterministic cuDNN settings for reproducible runs.

    Args:
        seed: Integer seed applied to Python, NumPy, PyTorch CPU, and all CUDA
            generators. cuDNN benchmarking is disabled so it does not select
            non-deterministic algorithms.

    Note:
        Complete reproducibility can still vary across PyTorch versions,
        operating systems, and GPU hardware.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_conversation(conversation):
    """Normalize a dataset conversation value to a list of message dictionaries.

    Args:
        conversation: A native list of messages, a JSON-encoded string
            containing that list, or another value.

    Returns:
        The parsed message list. Returns an empty list when the value is absent,
        malformed JSON, or not a list.
    """
    if isinstance(conversation, str):
        try:
            conversation = json.loads(conversation)
        except json.JSONDecodeError:
            return []
    return conversation if isinstance(conversation, list) else []


def conversation_input(conversation, input_mode):
    """Extract the requested text from one ``conversation_a`` record.

    Args:
        conversation: Raw ``conversation_a`` value from the dataset.
        input_mode: ``"user"`` for user messages, ``"assistant"`` for LLM
            messages, or ``"both"`` for both roles in their original order.

    Returns:
        Non-empty selected message contents joined with blank lines. In
        ``"both"`` mode, each message is prefixed with ``<user>`` or
        ``<assistant>`` to preserve role information. Returns an empty string
        when no eligible content is present.

    Note:
        This function deliberately never accesses ``conversation_b``.
    """
    parts = []
    for message in parse_conversation(conversation):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).lower()
        content = message.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        content = content.strip()

        if input_mode == "user" and role in USER_ROLES:
            parts.append(content)
        elif input_mode == "assistant" and role in ASSISTANT_ROLES:
            parts.append(content)
        elif input_mode == "both":
            if role in USER_ROLES:
                parts.append(f"<user> {content}")
            elif role in ASSISTANT_ROLES:
                parts.append(f"<assistant> {content}")
    return "\n\n".join(parts)


def load_examples(max_rows, input_mode):
    """Load labeled text examples from the first ``max_rows`` dataset records.

    Args:
        max_rows: Maximum number of training-split rows to inspect.
        input_mode: Message selection passed to :func:`conversation_input`.

    Returns:
        A list of ``(text, model_a_label)`` tuples from ``conversation_a``.

    Raises:
        RuntimeError: If no row supplies both selected text and a ``model_a``
            label. The Hugging Face dataset-loading exception is propagated for
            missing credentials, gated-access restrictions, or network errors.
    """
    dataset = load_dataset(DATASET_NAME, split="train", token=os.getenv("HF_TOKEN"))
    examples = []
    for row in dataset.select(range(min(max_rows, len(dataset)))):
        text = conversation_input(row.get("conversation_a"), input_mode)
        label = row.get("model_a")
        if text and label:
            examples.append((text, str(label)))
    if not examples:
        raise RuntimeError(f"No usable {input_mode!r} text was found in conversation_a.")
    return examples


class ResponseDataset(Dataset):
    """PyTorch dataset that stores fixed-length tokens and integer class labels.

    Tokenization is done once during construction rather than for every batch,
    which keeps ``__getitem__`` lightweight.

    Args:
        examples: Sequence of ``(text, string_label)`` pairs.
        tokenizer: Hugging Face tokenizer used to create token IDs and masks.
        label_to_id: Mapping from each string class label to its integer ID.
        max_length: Token count to retain per example. Longer inputs are
            truncated; shorter inputs are padded.

    Attributes:
        labels: Integer label ID for each example.
        tokens: Tokenizer output containing ``input_ids`` and
            ``attention_mask`` arrays.
    """

    def __init__(self, examples, tokenizer, label_to_id, max_length):
        """Tokenize all examples and convert their labels to integer IDs."""
        self.labels = [label_to_id[label] for _, label in examples]
        self.tokens = tokenizer(
            [text for text, _ in examples],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_attention_mask=True,
        )

    def __len__(self):
        """Return the number of tokenized examples available to the loader."""
        return len(self.labels)

    def __getitem__(self, index):
        """Return one example as PyTorch tensors.

        Args:
            index: Zero-based example position.

        Returns:
            A dictionary with ``input_ids``, ``attention_mask``, and ``labels``
            tensors suitable for the training and evaluation loops.
        """
        return {
            "input_ids": torch.tensor(self.tokens["input_ids"][index], dtype=torch.long),
            "attention_mask": torch.tensor(self.tokens["attention_mask"][index], dtype=torch.long),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


class TextCNN(nn.Module):
    """Convolutional text classifier using 3-, 4-, and 5-token feature windows.

    An embedding layer converts tokens to vectors. Parallel 1D convolutions
    detect local patterns, masked global max pooling retains the strongest
    valid feature from each filter, and a linear head predicts ``model_a``.

    Args:
        vocab_size: Number of tokenizer vocabulary entries.
        classes: Number of target ``model_a`` labels.
        pad_id: Token ID reserved for padding.
        embed_dim: Dimension of each learned token embedding.
        filters: Number of convolutional filters for each kernel width.
        dropout: Probability used before the classification layer.
    """

    def __init__(self, vocab_size, classes, pad_id, embed_dim=128, filters=128, dropout=0.25):
        """Initialize CNN layers and its classification head."""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, filters, k) for k in (3, 4, 5)])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(filters * 3, classes))

    def forward(self, input_ids, attention_mask):
        """Compute one logit vector per input with padding-aware max pooling.

        Args:
            input_ids: Integer token tensor with shape
                ``(batch_size, sequence_length)``.
            attention_mask: Binary validity mask with the same shape; one marks
                a real token and zero marks padding.

        Returns:
            Unnormalized class logits with shape ``(batch_size, classes)``.
        """
        embedded = self.embedding(input_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            values = F.gelu(conv(embedded))
            valid = F.max_pool1d(attention_mask.float().unsqueeze(1), conv.kernel_size[0], 1)
            values = values.masked_fill(valid == 0, torch.finfo(values.dtype).min)
            pooled.append(values.amax(dim=-1))
        return self.head(torch.cat(pooled, dim=1))


class TextGRU(nn.Module):
    """Bidirectional GRU text classifier that summarizes each full sequence.

    The forward and backward final GRU states are concatenated, passed through
    dropout, and classified into a ``model_a`` label. Packed sequences ensure
    that padding tokens do not influence the recurrent state.

    Args:
        vocab_size: Number of tokenizer vocabulary entries.
        classes: Number of target ``model_a`` labels.
        pad_id: Token ID reserved for padding.
        embed_dim: Dimension of each learned token embedding.
        hidden_dim: GRU hidden-state size in each direction.
        dropout: Probability used before the classification layer.
    """

    def __init__(self, vocab_size, classes, pad_id, embed_dim=128, hidden_dim=192, dropout=0.25):
        """Initialize the embedding, bidirectional GRU, and output head."""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * 2, classes))

    def forward(self, input_ids, attention_mask):
        """Compute class logits from the final state in both GRU directions.

        Args:
            input_ids: Integer token tensor with shape
                ``(batch_size, sequence_length)``.
            attention_mask: Binary validity mask with the same shape.

        Returns:
            Unnormalized class logits with shape ``(batch_size, classes)``.
        """
        lengths = attention_mask.sum(1).clamp_min(1).cpu()
        packed = pack_padded_sequence(
            self.embedding(input_ids), lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return self.head(torch.cat((hidden[-2], hidden[-1]), dim=1))


def evaluate(model, loader, device):
    """Evaluate a classifier on a held-out loader without computing gradients.

    Args:
        model: CNN or GRU classifier to evaluate.
        loader: DataLoader yielding token IDs, masks, and integer labels.
        device: PyTorch device that holds the model and receives batch tensors.

    Returns:
        Dictionary with the average cross-entropy ``loss``, classification
        ``accuracy``, and per-class-average ``macro_f1``.

    Note:
        The function switches the model to evaluation mode. The caller should
        call ``model.train()`` again before additional training.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    targets, predictions = [], []
    total_loss = 0.0

    with torch.inference_mode():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(ids, mask)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
    }


def print_metrics(model_name, epoch, metrics):
    """Print CNN or RNN validation metrics in the shared comparison format.

    Args:
        model_name: Display name, normally ``"CNN"`` or ``"RNN"``.
        epoch: One-based training epoch number.
        metrics: Dictionary returned by :func:`evaluate`.
    """
    print(
        f"{model_name} epoch {epoch}: "
        f"loss={metrics['loss']:.4f} "
        f"accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )


def print_epoch_examples(model_name, model, examples, tokenizer, label_to_id, device,
                         max_length, preview_chars):
    """Print selected validation inputs and this model's independent predictions.

    Args:
        model_name: Display name shown at the beginning of each output line.
        model: Trained classifier used to produce predictions.
        examples: ``(text, expected_string_label)`` pairs to display.
        tokenizer: Same tokenizer used when training the classifier.
        label_to_id: Mapping used to translate predicted IDs back to labels.
        device: Device on which to execute the prediction pass.
        max_length: Maximum token count accepted by the model.
        preview_chars: Maximum number of normalized input characters to print.

    Returns:
        ``None``. Nothing is printed when ``examples`` is empty.
    """
    if not examples:
        return

    model.eval()
    texts = [text for text, _ in examples]
    batch = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.inference_mode():
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        probabilities = logits.softmax(dim=1).cpu()
        prediction_ids = probabilities.argmax(dim=1).tolist()

    id_to_label = {index: label for label, index in label_to_id.items()}
    for number, ((text, expected), prediction_id, probs) in enumerate(
        zip(examples, prediction_ids, probabilities), start=1
    ):
        preview = " ".join(text.split())
        if len(preview) > preview_chars:
            preview = preview[:preview_chars].rstrip() + "..."
        predicted = id_to_label[prediction_id]
        confidence = probs[prediction_id].item()
        print(
            f"{model_name} sample {number}: input={preview!r} "
            f"expected={expected!r} predicted={predicted!r} "
            f"confidence={confidence:.4f}"
        )


def train_one_epoch(model, loader, optimizer, device):
    """Train one classifier for one complete pass over its training loader.

    Args:
        model: CNN or GRU classifier to update.
        loader: DataLoader yielding training batches.
        optimizer: Optimizer associated exclusively with ``model``.
        device: PyTorch device that holds the model and receives batch tensors.

    Returns:
        ``None``. Model weights are updated in place.

    The update uses cross-entropy loss, gradient clipping with maximum norm
    1.0, and the supplied optimizer. CNN and RNN calls are separate, so their
    parameters and optimizer states remain independent.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        loss = criterion(model(ids, mask), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def main():
    """Parse options, prepare data, train both classifiers, and save checkpoints.

    The CNN and bidirectional GRU use identical data splits and reporting
    fields, but each has its own model, optimizer, training pass, and
    evaluation result. Checkpoints are written as ``chatbot_arena_cnn.pt`` and
    ``chatbot_arena_rnn.pt`` in the current working directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--show-examples",
        type=int,
        default=1,
        help="held-out inputs and predictions to print per model after each epoch (0 disables)",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=300,
        help="maximum displayed characters per input preview",
    )
    parser.add_argument(
        "--input-mode",
        choices=("user", "assistant", "both"),
        default="assistant",
        help="conversation_a fields used as input (default: assistant)",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    examples = load_examples(args.rows, args.input_mode)
    labels = sorted({label for _, label in examples})
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_examples, eval_examples = train_test_split(
        examples,
        test_size=0.2,
        random_state=args.seed,
        stratify=[label for _, label in examples],
    )

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    train_set = ResponseDataset(train_examples, tokenizer, label_to_id, args.max_length)
    eval_set = ResponseDataset(eval_examples, tokenizer, label_to_id, args.max_length)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size)

    cnn = TextCNN(len(tokenizer), len(labels), tokenizer.pad_token_id or 0).to(device)
    rnn = TextGRU(len(tokenizer), len(labels), tokenizer.pad_token_id or 0).to(device)
    cnn_optimizer = torch.optim.AdamW(cnn.parameters(), lr=args.lr)
    rnn_optimizer = torch.optim.AdamW(rnn.parameters(), lr=args.lr)
    shown_examples = eval_examples[:max(args.show_examples, 0)]

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(cnn, train_loader, cnn_optimizer, device)
        print_metrics("CNN", epoch, evaluate(cnn, eval_loader, device))
        print_epoch_examples(
            "CNN", cnn, shown_examples, tokenizer, label_to_id, device,
            args.max_length, args.preview_chars,
        )

        train_one_epoch(rnn, train_loader, rnn_optimizer, device)
        print_metrics("RNN", epoch, evaluate(rnn, eval_loader, device))
        print_epoch_examples(
            "RNN", rnn, shown_examples, tokenizer, label_to_id, device,
            args.max_length, args.preview_chars,
        )

    metadata = {
        "labels": labels,
        "tokenizer": TOKENIZER_NAME,
        "max_length": args.max_length,
        "input_mode": args.input_mode,
        "source": "conversation_a only",
    }
    torch.save({"model_state": cnn.state_dict(), **metadata}, "chatbot_arena_cnn.pt")
    torch.save({"model_state": rnn.state_dict(), **metadata}, "chatbot_arena_rnn.pt")


if __name__ == "__main__":
    main()
