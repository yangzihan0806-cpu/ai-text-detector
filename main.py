# Chapter 6 of "Build a Large Language Model (From Scratch)" (Sebastian Raschka),
# "Fine-tuning for classification", applied to AI-text detection on HAP-E.
#
# The code below follows the chapter's own script (ch06/01_main-chapter-code/
# gpt_class_finetune.py in https://github.com/rasbt/LLMs-from-scratch,
# Apache License 2.0). The model, the frozen layers, the new 2-class output
# head, the loss and the training loop are exactly the book's.
#
# What changed compared with the book:
#   - data: the spam download / balancing / splitting steps are gone, because
#     data/train.csv.gz, validation.csv and test.csv are already balanced and
#     split (see prepare_hape.py). Label 0 = human, 1 = AI.
#   - max_length is fixed to 256 (the book uses the longest training text,
#     which here would be longer than GPT-2's 1024-token context).
#   - extras at the end: per-model / per-genre test accuracy, saving the model,
#     and classify_text() (the book's classify_review with new label names).
#
# Needs previous_chapters.py and gpt_download.py (copied from the book's repo)
# in the same folder.
#
#   python main.py --test_mode   # tiny random model on 200 texts: checks the code runs
#   python main.py --small       # GPT-2 124M on train_small.csv (2,000 texts)
#   python main.py               # GPT-2 124M on the full training set
#   python main.py --small --eval_only   # no training: reload the saved .pth, evaluate only
#
# Early stopping: after every epoch the loss on the full validation set is
# computed; the best epoch's weights are saved (and used for the final
# evaluation), and training stops after --patience epochs without improvement.
# The model is saved during training (ai_text_classifier_small.pth for
# --small, ai_text_classifier.pth for the full run), and every printed line is
# also appended to run_log.txt.

import argparse
import time

import matplotlib.pyplot as plt
import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

from previous_chapters import GPTModel, load_weights_into_gpt


########################################
# Dataset (book: SpamDataset, unchanged)
########################################

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        return max(len(encoded_text) for encoded_text in self.encoded_texts)


########################################
# Loss / accuracy / training loop (book, unchanged)
########################################

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # Logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, save_path=None, patience=2):
    # Initialize lists to track losses and examples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    # New (not in the book): early stopping on the loss over the FULL validation set
    best_val_loss, best_epoch, best_state = float("inf"), 0, None

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            examples_seen += input_batch.shape[0]  # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

        # New: keep the weights from the epoch with the lowest full validation loss
        full_val_loss = calc_loss_loader(val_loader, model, device)
        model.train()
        if full_val_loss < best_val_loss:
            best_val_loss, best_epoch = full_val_loss, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
            print(f"  full val loss {full_val_loss:.4f}  <- best so far"
                  + (f", saved to {save_path}" if save_path else ""))
        else:
            print(f"  full val loss {full_val_loss:.4f}  (best {best_val_loss:.4f} at epoch {best_epoch})")
            if epoch + 1 - best_epoch >= patience:
                print(f"  no improvement for {patience} epochs -> early stopping")
                break

    # go back to the best weights before the final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Using the weights from epoch {best_epoch} (full val loss {best_val_loss:.4f})")

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for examples seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{label}-plot.pdf")
    # plt.show()


########################################
# Using the model (book: classify_review; labels renamed)
########################################

def classify_text(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    assert max_length is not None, (
        "max_length must be specified. If you want to use the full model context, "
        "pass max_length=model.pos_emb.weight.shape[0]."
    )
    assert max_length <= supported_context_length, (
        f"max_length ({max_length}) exceeds model's supported context length ({supported_context_length})."
    )
    # Truncate sequences if they are too long
    input_ids = input_ids[:max_length]

    # No padding here (changed from the book): every training text was longer than
    # max_length, so the model never saw <|endoftext|> padding. With padding, the
    # last position (whose logits we read) would be a pad token -> meaningless output.
    # A single text needs no padding anyway, since there is no batch to line up.
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)  # add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "AI" if predicted_label == 1 else "human"


########################################
# Extra (not in the book): accuracy per generating model and per genre
########################################

def accuracy_by_group(dataset, data_loader, model, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for input_batch, _ in data_loader:  # test_loader is not shuffled, so order = CSV order
            logits = model(input_batch.to(device))[:, -1, :]
            preds.extend(torch.argmax(logits, dim=-1).tolist())
    df = dataset.data.copy()
    df["correct"] = (pd.Series(preds, index=df.index) == df["Label"])
    print("\nTest accuracy by source (chunk_2 = human texts):")
    print((df.groupby("Source")["correct"].mean() * 100).round(2).to_string())
    print("\nTest accuracy by genre:")
    print((df.groupby("Genre")["correct"].mean() * 100).round(2).to_string())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Finetune a GPT model to tell human from AI text (HAP-E)"
    )
    parser.add_argument("--test_mode", default=False, action="store_true",
                        help="Tiny random GPT on a few texts, just to check that the code runs.")
    parser.add_argument("--small", default=False, action="store_true",
                        help="Train on data/train_small.csv (2,000 texts) instead of the full set.")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--patience", type=int, default=2,
                        help="Early stopping: stop after this many epochs without a lower full validation loss.")
    parser.add_argument("--eval_only", default=False, action="store_true",
                        help="Skip training: load the saved .pth and only evaluate.")
    args = parser.parse_args()

    # everything printed also goes to run_log.txt, so closing the window loses nothing
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, s):
            for st in self.streams:
                st.write(s)
                st.flush()
        def flush(self):
            for st in self.streams:
                st.flush()
    import sys
    sys.stdout = Tee(sys.__stdout__, open("run_log.txt", "a", encoding="utf-8"))
    print("\n" + "=" * 60 + f"\n{time.strftime('%Y-%m-%d %H:%M:%S')}  python main.py {' '.join(sys.argv[1:])}")

    model_path = "ai_text_classifier_small.pth" if args.small else "ai_text_classifier.pth"

    ########################################
    # Create data loaders
    ########################################
    tokenizer = tiktoken.get_encoding("gpt2")

    train_file = "data/train_small.csv" if (args.small or args.test_mode) else "data/train.csv.gz"

    train_dataset = SpamDataset(
        csv_file=train_file,
        max_length=args.max_length,
        tokenizer=tokenizer
    )

    val_dataset = SpamDataset(
        csv_file="data/validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    test_dataset = SpamDataset(
        csv_file="data/test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    if args.test_mode:  # keep it tiny
        for ds, n in [(train_dataset, 200), (val_dataset, 40), (test_dataset, 40)]:
            ds.data = ds.data.iloc[:n].reset_index(drop=True)
            ds.encoded_texts = ds.encoded_texts[:n]

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")

    ########################################
    # Load pretrained model
    ########################################

    # Small GPT model for testing purposes
    if args.test_mode:
        BASE_CONFIG = {
            "vocab_size": 50257,
            "context_length": args.max_length,
            "drop_rate": 0.0,
            "qkv_bias": False,
            "emb_dim": 12,
            "n_layers": 1,
            "n_heads": 2
        }
        model = GPTModel(BASE_CONFIG)
        model.eval()
        device = "cpu"

    # Code as it is used in the main chapter
    else:
        CHOOSE_MODEL = "gpt2-small (124M)"

        BASE_CONFIG = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 1024,  # Context length
            "drop_rate": 0.0,        # Dropout rate
            "qkv_bias": True         # Query-key-value bias
        }

        model_configs = {
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }

        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

        assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
            f"Dataset length {train_dataset.max_length} exceeds model's context "
            f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
            f"`max_length={BASE_CONFIG['context_length']}`"
        )

        model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
        from gpt_download import download_and_load_gpt2

        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

        model = GPTModel(BASE_CONFIG)
        load_weights_into_gpt(model, params)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print("Device:", device)

    ########################################
    # Modify the pretrained model
    ########################################

    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(123)

    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
    model.to(device)

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True

    ########################################
    # Finetune modified model
    ########################################

    if args.eval_only:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded fine-tuned weights from {model_path} (no training)")
    else:
        start_time = time.time()
        torch.manual_seed(123)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

        num_epochs = args.num_epochs
        train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs, eval_freq=50, eval_iter=5,
            save_path=None if args.test_mode else model_path, patience=args.patience,
        )
        num_epochs = len(train_accs)  # may be fewer than requested if early stopping kicked in

        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        print(f"Training completed in {execution_time_minutes:.2f} minutes.")

        ########################################
        # Plot results
        ########################################

        # loss plot
        epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
        examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
        plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

        # accuracy plot
        epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
        examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
        plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")


    ########################################
    # Evaluate on the full data sets
    ########################################

    # test first (most important), the full training set last (slowest)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    accuracy_by_group(test_dataset, test_loader, model, device)

    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    print(f"\nValidation accuracy: {val_accuracy*100:.2f}%")
    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    print(f"Training accuracy: {train_accuracy*100:.2f}%")

    ########################################
    # Try the model on two new texts
    ########################################
    print()

    text_1 = (
        "The rapid advancement of artificial intelligence has transformed numerous "
        "industries, offering unprecedented opportunities for innovation while also "
        "raising important ethical considerations that society must carefully address."
    )
    text_2 = (
        "honestly i didn't think the bus would be that late again, so i just walked. "
        "got there soaked and the lecture had already started lol"
    )
    for t in (text_1, text_2):
        print(classify_text(t, model, tokenizer, device, max_length=train_dataset.max_length), "<-", t[:60], "...")
