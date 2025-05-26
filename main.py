import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import os
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define constants
KEY_MAPPINGS = {
    "w+a": 0, "w": 1, "w+d": 2, "a": 3,
    "None": 4, "d": 5,"s+a": 6,"s":7,"s+d": 8
}
NUM_KEYS = len(KEY_MAPPINGS)
REVERSE_KEY_MAPPINGS = {v: k for k, v in KEY_MAPPINGS.items()}

class MovementDataset(Dataset):
    def __init__(self, data_path: str, sequence_length: int = 20):
        """
        Dataset for loading and processing movement data
        
        Args:
            data_path: Path to JSON file containing movement sequences
            sequence_length: Number of previous actions to use as input
        """
        self.sequence_length = sequence_length
        self.sequences = []
        
        # Load data
        with open(data_path, 'r') as f:
            data = json.load(f)
            
        # Process each sequence
        for seq in data:
            actions = seq['actions']
            if len(actions) < sequence_length + 1:
                continue
                
            # Create sliding windows
            for i in range(len(actions) - sequence_length):
                input_keys = [KEY_MAPPINGS[a['key']] for a in actions[i:i+sequence_length]]
                input_durations = [a['duration'] for a in actions[i:i+sequence_length]]
                target_key = KEY_MAPPINGS[actions[i+sequence_length]['key']]
                target_duration = actions[i+sequence_length]['duration']
                
                self.sequences.append({
                    'input_keys': input_keys,
                    'input_durations': input_durations,
                    'target_key': target_key,
                    'target_duration': target_duration
                })
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return (
            torch.tensor(seq['input_keys'], dtype=torch.long),
            torch.tensor(seq['input_durations'], dtype=torch.float32),
            torch.tensor(seq['target_key'], dtype=torch.long),
            torch.tensor(seq['target_duration'], dtype=torch.float32)
        )

class MovementPredictor(nn.Module):
    def __init__(self, hidden_size: int = 128, num_keys: int = NUM_KEYS, num_layers: int = 2):
        '''        
        Args:
            hidden_size: Size of hidden layers
            num_keys: Number of possible key combinations
            num_layers: Number of LSTM/GRU layers
        '''
        super(MovementPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_keys = num_keys
        self.num_layers = num_layers
        
        self.key_embedding = nn.Embedding(num_keys, hidden_size)
        
        self.encoder = nn.GRU(
            input_size=hidden_size + 1,  # +1 for duration
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        self.decoder = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.key_classifier = nn.Linear(hidden_size, num_keys)
        self.duration_predictor = nn.Linear(hidden_size, 1)
        
    def forward(self, keys, durations, hidden=None):
        batch_size = keys.size(0)
        seq_len = keys.size(1)
        
        # Embed keys
        key_embedded = self.key_embedding(keys)  # [batch, seq_len, hidden_size]
        
        # Concatenate with durations
        encoder_input = torch.cat([key_embedded, durations.unsqueeze(-1)], dim=-1)
        
        # Encode sequence
        encoder_output, hidden = self.encoder(encoder_input, hidden)  # [batch, seq_len, 2*hidden_size]
        
        # Apply attention
        attention_scores = self.attention(encoder_output).squeeze(-1)  # [batch, seq_len]
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(1)  # [batch, 1, seq_len]
        
        # Context vector
        context = torch.bmm(attention_weights, encoder_output)  # [batch, 1, 2*hidden_size]
        
        # Decode
        decoder_output, hidden = self.decoder(context, None)  # [batch, 1, hidden_size]
        
        # Predict next key and duration
        key_logits = self.key_classifier(decoder_output.squeeze(1))  # [batch, num_keys]
        duration = F.relu(self.duration_predictor(decoder_output.squeeze(1)))  # [batch, 1]
        
        return key_logits, duration, hidden
    
    def generate_sequence(self, seed_keys, seed_durations, length=20):
        """
        Generate a movement sequence from a seed sequence
        
        Args:
            seed_keys: Initial key sequence [seq_len]
            seed_durations: Initial duration sequence [seq_len]
            length: Length of sequence to generate
            
        Returns:
            Generated key sequence and durations
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensors and add batch dimension
            keys = torch.tensor(seed_keys, dtype=torch.long).unsqueeze(0)
            durations = torch.tensor(seed_durations, dtype=torch.float32).unsqueeze(0)
            
            generated_keys = seed_keys.copy()
            generated_durations = seed_durations.copy()
            
            for _ in range(length):
                # Get prediction
                key_logits, duration, _ = self(keys, durations)
                
                # Sample next key from logits
                probs = F.softmax(key_logits, dim=-1)
                next_key = torch.multinomial(probs, 1).item()
                
                # Get predicted duration
                next_duration = duration.item()
                
                # Append to generated sequence
                generated_keys.append(next_key)
                generated_durations.append(next_duration)
                
                # Update input for next prediction
                keys = torch.tensor([generated_keys[-len(seed_keys):]], dtype=torch.long)
                durations = torch.tensor([generated_durations[-len(seed_durations):]], dtype=torch.float32)
            
        return generated_keys[len(seed_keys):], generated_durations[len(seed_durations):]

def train_model(model, train_loader, val_loader, epochs=40, learning_rate=0.001, weight_decay=1e-4):
    """
    Train the movement prediction model
    
    Args:
        model: MovementPredictor model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        
    Returns:
        Lists of training and validation losses
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    key_criterion = nn.CrossEntropyLoss()
    duration_criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        
        for keys, durations, target_key, target_duration in train_loader:
            keys = keys.to(device)
            durations = durations.to(device)
            target_key = target_key.to(device)
            target_duration = target_duration.to(device)
            
            # Forward pass
            key_logits, pred_duration, _ = model(keys, durations)
            
            # Calculate loss
            key_loss = key_criterion(key_logits, target_key)
            duration_loss = duration_criterion(pred_duration.squeeze(), target_duration)
            loss = 0.7 * key_loss + 0.3 * duration_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        
        with torch.no_grad():
            for keys, durations, target_key, target_duration in val_loader:
                keys = keys.to(device)
                durations = durations.to(device)
                target_key = target_key.to(device)
                target_duration = target_duration.to(device)
                
                # Forward pass
                key_logits, pred_duration, _ = model(keys, durations)
                
                # Calculate loss
                key_loss = key_criterion(key_logits, target_key)
                duration_loss = duration_criterion(pred_duration.squeeze(), target_duration)
                loss = 0.7 * key_loss + 0.3 * duration_loss
                
                epoch_val_loss += loss.item()
            
        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_movement_model.pth")
    
    return train_losses, val_losses


def plot_training_history(train_losses, val_losses):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.close()

def analyze_generated_sequence(keys, durations):
    """Analyze and print statistics about a generated movement sequence"""
    key_names = [REVERSE_KEY_MAPPINGS[k] for k in keys]
    
    print("Generated Sequence Analysis:")
    print(f"Total length: {len(keys)} actions")
    
    # Key distribution
    key_counts = {}
    for k in key_names:
        key_counts[k] = key_counts.get(k, 0) + 1
    
    print("\nKey Distribution:")
    for k, count in sorted(key_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {count} ({count/len(keys)*100:.1f}%)")
    
    # Duration statistics
    print(f"\nDuration Statistics:")
    print(f"Average duration: {sum(durations)/len(durations):.2f}s")
    print(f"Min duration: {min(durations):.2f}s")
    print(f"Max duration: {max(durations):.2f}s")
    
    # Patterns
    print("\nSample sequence (first 10 actions):")
    for i in range(min(10, len(keys))):
        print(f"{key_names[i]}: {durations[i]:.2f}s")

def main():
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create dummy data if it doesn't exist
    data_path = "movement_data.json"
    if not os.path.exists(data_path):
        data_path = create_dummy_data()
    
    # Create dataset
    dataset = MovementDataset(data_path, sequence_length=20)
    
    # Split into train and validation sets
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = MovementPredictor(hidden_size=128, num_keys=NUM_KEYS, num_layers=2)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, epochs=40, learning_rate=0.001
    )
    
    plot_training_history(train_losses, val_losses)
    
    model.load_state_dict(torch.load("best_movement_model.pth"))

    seed_keys = [KEY_MAPPINGS["w"], KEY_MAPPINGS["a"], KEY_MAPPINGS["s"], KEY_MAPPINGS["d"],KEY_MAPPINGS["w+a"]]
    seed_durations = [1.2, 0.7, 0.5, 0.9, 0.2]
    
    generated_keys, generated_durations = model.generate_sequence(
        seed_keys, seed_durations, length=40
    )
    
    # Analyze generated sequence
    analyze_generated_sequence(generated_keys, generated_durations)
    
    # Save a longer generated sequence to file
    long_keys, long_durations = model.generate_sequence(
        seed_keys, seed_durations, length=300
    )
    
    sequence_data = {
        "generated_sequence": [
            {"key": REVERSE_KEY_MAPPINGS[k], "duration": d}
            for k, d in zip(long_keys, long_durations)
        ]
    }
    
    with open("generated_movement.json", 'w') as f:
        json.dump(sequence_data, f, indent=2)
    
    print("\nGenerated longer movement sequence and saved to 'generated_movement.json'")

if __name__ == "__main__":
    main()