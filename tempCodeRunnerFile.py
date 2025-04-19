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