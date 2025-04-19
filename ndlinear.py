import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt
import time
# --- Tokenization and Vocabulary ---

def tokenize(text):
    """A simple tokenizer that lowercases and splits the text by whitespace."""
    return text.lower().split()

def build_vocab(texts, min_freq=1):
    """
    Build a vocabulary from a list of texts.
    Words appearing less than min_freq times are ignored.
    """
    freq = {}
    for text in texts:
        for token in tokenize(text):
            freq[token] = freq.get(token, 0) + 1
    # Start with special tokens: <PAD> for padding and <UNK> for unknown words
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, count in freq.items():
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab

def text_to_indices(text, vocab, max_len=20):
    """
    Convert a text to a list of token indices, padding or truncating to max_len.
    """
    tokens = tokenize(text)
    indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    # Pad with <PAD> token index if the list is too short
    if len(indices) < max_len:
        indices = indices + [vocab["<PAD>"]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices

# --- PyTorch Dataset for AG News ---

class AGNewsDataset(Dataset):
    def __init__(self, split, vocab, max_len=20):
        dataset = load_dataset("ag_news", split=split)
        self.texts = dataset["text"]
        self.labels = dataset["label"]
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        indices = text_to_indices(text, self.vocab, self.max_len)
        # Return tensor versions of the indices and label
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)
if __name__ == "__main__":
# Build vocabulary using the first 1000 texts (for speed, as an example)
    ag_train = load_dataset("ag_news", split="train")
    sample_texts = ag_train["text"][:1000]
    vocab = build_vocab(sample_texts, min_freq=2)
    print("Vocabulary size:", len(vocab))
    # --- Model using standard nn.Linear ---

    class BaselineModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
            super(BaselineModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.linear = nn.Linear(embedding_dim, hidden_dim)
            self.classifier = nn.Linear(hidden_dim, num_classes)
            
        def forward(self, x):
            # x shape: (batch_size, max_len)
            x = self.embedding(x)  # shape becomes (batch_size, max_len, embedding_dim)
            # We need to apply nn.Linear on the last dimension, so we reshape:
            batch_size, seq_len, emb_dim = x.shape
            x = x.view(batch_size * seq_len, emb_dim)  # flatten batch and seq length
            x = self.linear(x)  # now x has shape (batch_size * seq_len, hidden_dim)
            # Reshape back to separate batch and sequence dimensions:
            x = x.view(batch_size, seq_len, hidden_dim)
            # Average over the sequence dimension to get one vector per sample:
            x = x.mean(dim=1)  # shape: (batch_size, hidden_dim)
            logits = self.classifier(x)
            return logits

    # --- Model using NdLinear ---
    # Make sure to import NdLinear; adjust the import if your folder structure differs.
    try:
        from NdLinear.ndlinear import NdLinear
    except ImportError:
        import sys
        # Adjust the following path to the location where NdLinear is cloned
        sys.path.append("/Users/raneemmousa/Desktop/NdLinear/NdLinear")
        from NdLinear import NdLinear

    class NdModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
            super(NdModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.ndlinear = NdLinear([embedding_dim], [hidden_dim])
            self.classifier = nn.Linear(hidden_dim, num_classes)
            
        def forward(self, x):
            # x shape: (batch_size, max_len)
            x = self.embedding(x)  # shape: (batch_size, max_len, embedding_dim)
            # NdLinear can work on multi-dimensional tensors without reshaping
            x = self.ndlinear(x)  # shape becomes (batch_size, max_len, hidden_dim)
            # Average over the sequence dimension:
            x = x.mean(dim=1)
            logits = self.classifier(x)
            return logits

    # Define model hyperparameters:
    embedding_dim = 50
    hidden_dim = 32
    num_classes = 4
    max_len = 20
    # --- Setting Up DataLoader and Training Loop ---

    batch_size = 32

    # Create the training dataset and a DataLoader
    train_dataset = AGNewsDataset("train", vocab, max_len=max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Set device: use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Choose one model to train; try switching between BaselineModel and NdModel for comparison.
    model = BaselineModel(len(vocab), embedding_dim, hidden_dim, num_classes).to(device)
    # To use NdModel, comment out the above line and uncomment the next:
    # model = NdModel(len(vocab), embedding_dim, hidden_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop for one epoch (for demonstration)
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            """ if (batch_idx + 1) % 100 == 0:  # Print every 100 batches
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {running_loss/100:.4f}")
                running_loss = 0.0"""

    print("Training complete.")
    import time

    def train_model(model, train_loader, criterion, optimizer, device, num_epochs=5):
        model.to(device)
        model.train()
        
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                num_batches += 1
            
            average_loss = running_loss / len(train_loader)
            
            total_loss += running_loss
        
        elapsed_time = time.time() - start_time
        overall_average_loss = total_loss / num_batches
        return overall_average_loss, elapsed_time

    # Usage example, assuming other variables (train_loader, criterion, device, etc.) are set up:
    # Hyperparameters (as defined previously)
    embedding_dim = 50
    hidden_dim = 32
    num_classes = 4
    max_len = 20
    batch_size = 32
    num_epochs = 5  # Increase this if you want a more thorough comparison

    # Create the DataLoader as before
    train_dataset = AGNewsDataset("train", vocab, max_len=max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1. BaselineModel with nn.Linear
    baseline_model = BaselineModel(len(vocab), embedding_dim, hidden_dim, num_classes)
    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)
    baseline_loss, baseline_time = train_model(baseline_model, train_loader, nn.CrossEntropyLoss(), baseline_optimizer, device, num_epochs=num_epochs)
    print(f"BaselineModel - Final Average Loss: {baseline_loss:.4f}, Training Time: {baseline_time:.2f} sec")

    # 2. NdModel with NdLinear
    nd_model = NdModel(len(vocab), embedding_dim, hidden_dim, num_classes)
    nd_optimizer = optim.Adam(nd_model.parameters(), lr=0.001)
    nd_loss, nd_time = train_model(nd_model, train_loader, nn.CrossEntropyLoss(), nd_optimizer, device, num_epochs=num_epochs)
    print(f"NdModel - Final Average Loss: {nd_loss:.4f}, Training Time: {nd_time:.2f} sec")
    from torch.utils.data import random_split

    # Assuming train_dataset is already created (AGNewsDataset)
    total_size = len(train_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)




    def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
        train_losses = []  # List to store training loss per epoch
        val_losses = []    # List to store validation loss per epoch
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            # Training loop for one epoch
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()         # Clear previous gradients
                outputs = model(inputs)         # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                loss.backward()                 # Backpropagate
                optimizer.step()                # Update parameters
                running_loss += loss.item()
            
            # Compute the average training loss for this epoch
            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Evaluate on the validation set
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        return train_losses, val_losses
    # Set number of epochs for the experiment
    num_epochs = 10

    # --- Train and Evaluate BaselineModel (using nn.Linear) ---
    baseline_model = BaselineModel(len(vocab), embedding_dim, hidden_dim, num_classes).to(device)
    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)
    print("\nTraining BaselineModel (nn.Linear)...")
    baseline_train_losses, baseline_val_losses = train_and_evaluate(baseline_model, train_loader, val_loader, nn.CrossEntropyLoss(), baseline_optimizer, device, num_epochs)

    # --- Train and Evaluate NdModel (using NdLinear) ---
    nd_model = NdModel(len(vocab), embedding_dim, hidden_dim, num_classes).to(device)
    nd_optimizer = optim.Adam(nd_model.parameters(), lr=0.001)
    print("\nTraining NdModel (NdLinear)...")
    nd_train_losses, nd_val_losses = train_and_evaluate(nd_model, train_loader, val_loader, nn.CrossEntropyLoss(), nd_optimizer, device, num_epochs)
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    # Plot training losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, baseline_train_losses, label='Baseline Train Loss', marker='o')
    plt.plot(epochs, nd_train_losses, label='NdModel Train Loss', marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot validation losses
    plt.subplot(1, 2, 2)
    plt.plot(epochs, baseline_val_losses, label='Baseline Val Loss', marker='o')
    plt.plot(epochs, nd_val_losses, label='NdModel Val Loss', marker='o')
    plt.title('Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()