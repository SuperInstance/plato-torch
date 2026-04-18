"""
InstinctNet + PolicyNet — neural networks for room instinct.

When PyTorch is available, these replace the statistical model
with real neural training. When it's not, TorchRoom falls back
to statistical tables (which still work well with enough data).
"""

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    
    class InstinctNet(nn.Module):
        """Value network: state → estimated value.
        
        Takes a state embedding and returns a scalar value estimate.
        "How good does this state feel?" — trained from outcomes.
        """
        
        def __init__(self, state_dim: int = 256, hidden_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Tanh()  # Output in [-1, 1]
            )
        
        def forward(self, state_embedding):
            return self.net(state_embedding)
    
    
    class PolicyNet(nn.Module):
        """Policy network: state → action distribution.
        
        "Given this state, which action should I take?"
        Returns logits over possible actions.
        """
        
        def __init__(self, state_dim: int = 256, num_actions: int = 10,
                     hidden_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_actions)
            )
        
        def forward(self, state_embedding):
            return self.net(state_embedding)  # raw logits
    
    
    class StrategyMeshNet(nn.Module):
        """Multi-agent strategy mesh network.
        
        Takes embeddings from multiple agents and predicts
        their coordination quality. "How well do these strategies mesh?"
        
        This is the network that goes beyond step-wise logic.
        It learns pattern-level synergy between agents.
        """
        
        def __init__(self, agent_dim: int = 64, num_agents: int = 4,
                     hidden_dim: int = 128):
            super().__init__()
            # Each agent gets its own embedding processor
            self.agent_encoder = nn.Sequential(
                nn.Linear(agent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, agent_dim)
            )
            
            # Cross-agent attention: how do agents' strategies interact?
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=agent_dim,
                num_heads=4,
                batch_first=True
            )
            
            # Output: synergy score
            self.synergy_head = nn.Sequential(
                nn.Linear(agent_dim * num_agents, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Tanh()
            )
        
        def forward(self, agent_embeddings):
            """
            Args:
                agent_embeddings: [batch, num_agents, agent_dim]
            Returns:
                synergy_score: [batch, 1]
            """
            # Encode each agent
            encoded = self.agent_encoder(agent_embeddings)
            
            # Cross-attention: agents "look at" each other
            attn_out, _ = self.cross_attention(encoded, encoded, encoded)
            
            # Concatenate all agent representations
            combined = attn_out.reshape(attn_out.size(0), -1)
            
            # Predict synergy
            return self.synergy_head(combined)
    
else:
    # Stubs when PyTorch not available
    class InstinctNet:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not installed. pip install torch")
    
    class PolicyNet:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not installed. pip install torch")
    
    class StrategyMeshNet:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not installed. pip install torch")
