"""
Self-Pruning Neural Network Implementation
Case Study for Tredence AI Engineering Intern
Author: Pradish G
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# ============================================================================
# PART 1: PRUNABLE LINEAR LAYER
# ============================================================================

class PrunableLinear(nn.Module):
    """
    Custom Linear layer with learnable gates for self-pruning.
    
    Each weight is multiplied by a gate value (0-1) that the network learns.
    Gates that approach 0 effectively "prune" their corresponding weights.
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Gate scores - learnable parameters that control sparsity
        # Initialized to ones (all weights active initially)
        self.gate_scores = nn.Parameter(torch.ones(out_features, in_features))
    
    def forward(self, x):
        """
        Forward pass with gating mechanism.
        
        Steps:
        1. Convert gate_scores to gates (0-1) using sigmoid
        2. Multiply weights element-wise with gates
        3. Perform standard linear operation
        """
        # Apply sigmoid to gate_scores to get values between 0 and 1
        gates = torch.sigmoid(self.gate_scores)
        
        # Element-wise multiplication: pruned_weights = weight * gates
        pruned_weights = self.weight * gates
        
        # Standard linear transformation with pruned weights
        return torch.nn.functional.linear(x, pruned_weights, self.bias)
    
    def get_sparsity(self, threshold=1e-2):
        """
        Calculate sparsity level: percentage of gates below threshold.
        """
        gates = torch.sigmoid(self.gate_scores)
        pruned = (gates < threshold).sum().item()
        total = gates.numel()
        return (pruned / total) * 100
    
    def get_gate_values(self):
        """Return the gate values for analysis."""
        return torch.sigmoid(self.gate_scores).detach().cpu().numpy()


# ============================================================================
# PART 2: NEURAL NETWORK ARCHITECTURE
# ============================================================================

class SelfPruningNetwork(nn.Module):
    """
    Feed-forward neural network using PrunableLinear layers.
    Designed for CIFAR-10 image classification (32x32 images, 10 classes).
    """
    
    def __init__(self):
        super(SelfPruningNetwork, self).__init__()
        
        # Flatten input: 3 channels * 32 * 32 = 3072
        self.fc1 = PrunableLinear(3072, 512)
        self.relu1 = nn.ReLU()
        
        self.fc2 = PrunableLinear(512, 256)
        self.relu2 = nn.ReLU()
        
        self.fc3 = PrunableLinear(256, 128)
        self.relu3 = nn.ReLU()
        
        self.fc4 = PrunableLinear(128, 10)  # 10 classes for CIFAR-10
    
    def forward(self, x):
        """Forward pass through the network."""
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.fc1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        x = self.relu3(x)
        
        x = self.fc4(x)
        return x
    
    def get_total_sparsity(self, threshold=1e-2):
        """Calculate overall network sparsity."""
        total_gates = 0
        pruned_gates = 0
        
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                total_gates += gates.numel()
                pruned_gates += (gates < threshold).sum().item()
        
        return (pruned_gates / total_gates) * 100 if total_gates > 0 else 0
    
    def get_all_gates(self):
        """Return all gate values for visualization."""
        all_gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                all_gates.extend(module.get_gate_values().flatten())
        return np.array(all_gates)


# ============================================================================
# PART 3: TRAINING AND EVALUATION
# ============================================================================

def train_network(model, train_loader, test_loader, num_epochs, lambda_sparsity, device):
    """
    Train the self-pruning network.
    
    Loss = CrossEntropyLoss + lambda * SparsityLoss
    where SparsityLoss = sum of all gate values (L1 norm)
    """
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    test_accuracies = []
    
    print(f"\n{'='*70}")
    print(f"Training with Lambda = {lambda_sparsity}")
    print(f"{'='*70}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        classification_loss_sum = 0
        sparsity_loss_sum = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Classification loss
            classification_loss = criterion(output, target)
            
            # Sparsity loss: L1 norm of all gates
            sparsity_loss = 0
            for module in model.modules():
                if isinstance(module, PrunableLinear):
                    gates = torch.sigmoid(module.gate_scores)
                    sparsity_loss += gates.sum()
            
            # Total loss
            total_loss_batch = classification_loss + lambda_sparsity * sparsity_loss
            
            # Backward pass
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            classification_loss_sum += classification_loss.item()
            sparsity_loss_sum += sparsity_loss.item()
        
        # Evaluation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        test_accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        train_losses.append(avg_loss)
        test_accuracies.append(test_accuracy)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Test Acc: {test_accuracy:.2f}% | "
                  f"Sparsity: {model.get_total_sparsity():.2f}%")
    
    return train_losses, test_accuracies


def evaluate_model(model, test_loader, device, lambda_value):
    """Evaluate model and collect results."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_accuracy = 100 * correct / total
    sparsity = model.get_total_sparsity()
    
    return {
        'lambda': lambda_value,
        'test_accuracy': test_accuracy,
        'sparsity': sparsity,
        'model': model
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    num_epochs = 20
    batch_size = 128
    
    # Load CIFAR-10 dataset
    print("\nLoading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Test with different lambda values
    lambda_values = [0.0001, 0.001, 0.01]
    results = []
    models = {}
    
    for lambda_val in lambda_values:
        print(f"\n\n{'#'*70}")
        print(f"# Training with Lambda = {lambda_val}")
        print(f"{'#'*70}")
        
        model = SelfPruningNetwork().to(device)
        
        # Train the model
        train_losses, test_accuracies = train_network(
            model, train_loader, test_loader, num_epochs, lambda_val, device
        )
        
        # Evaluate
        result = evaluate_model(model, test_loader, device, lambda_val)
        results.append(result)
        models[lambda_val] = (model, train_losses, test_accuracies)
        
        print(f"\nFinal Results for Lambda = {lambda_val}")
        print(f"  Test Accuracy: {result['test_accuracy']:.2f}%")
        print(f"  Sparsity Level: {result['sparsity']:.2f}%")
    
    # Print results table
    print(f"\n\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Lambda':<15} {'Test Accuracy (%)':<20} {'Sparsity (%)':<20}")
    print(f"{'-'*70}")
    for result in results:
        print(f"{result['lambda']:<15.6f} {result['test_accuracy']:<20.2f} {result['sparsity']:<20.2f}")
    
    # Visualization: Gate distribution for best model
    best_result = max(results, key=lambda x: x['sparsity'])
    best_model = best_result['model']
    best_gates = best_model.get_all_gates()
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Gate value distribution
    plt.subplot(1, 2, 1)
    plt.hist(best_gates, bins=100, edgecolor='black', alpha=0.7)
    plt.xlabel('Gate Value')
    plt.ylabel('Frequency')
    plt.title(f'Gate Value Distribution (Lambda = {best_result["lambda"]})')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Sigmoid midpoint')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Sparsity vs Lambda
    plt.subplot(1, 2, 2)
    lambdas = [r['lambda'] for r in results]
    sparsities = [r['sparsity'] for r in results]
    accuracies = [r['test_accuracy'] for r in results]
    
    ax1 = plt.gca()
    ax1.set_xlabel('Lambda')
    ax1.set_ylabel('Sparsity (%)', color='tab:blue')
    ax1.plot(lambdas, sparsities, 'o-', color='tab:blue', label='Sparsity', linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Test Accuracy (%)', color='tab:orange')
    ax2.plot(lambdas, accuracies, 's-', color='tab:orange', label='Accuracy', linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    plt.title('Sparsity vs Accuracy Trade-off')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pruning_results.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved as 'pruning_results.png'")
    
    return results, models


if __name__ == '__main__':
    results, models = main()
    print("\n✅ Training completed successfully!")
