import torch
import torch.nn as nn

class MTPHead(nn.Module):
    """
    Minimal Multi-Token Prediction head.
    Predicts 'mtp_tokens' future tokens beyond the immediate next token.
    """
    def __init__(self, d_model, vocab_size, mtp_tokens=2):
        super().__init__()
        self.mtp_tokens = mtp_tokens
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, hidden, mtp_labels=None):
        """
        hidden: [batch, seq_len, d_model]
        We can shift hidden states or do a small projection to predict future tokens.
        For simplicity, we just do a naive approach here.
        """
        batch_size, seq_len, d_model = hidden.size()
        # We'll predict the next 'mtp_tokens' tokens from the final hidden state
        logits = self.lm_head(hidden)  # [batch, seq_len, vocab_size]

        loss = None
        if mtp_labels is not None:
            # We'll assume mtp_labels have shape [batch, seq_len, mtp_tokens]
            # and we do a separate CE for each future token
            # This is again a simplified approach for demonstration
            vocab_size = logits.size(-1)
            total_loss = 0.0
            for i in range(self.mtp_tokens):
                # For position j in seq_len, the label is mtp_labels[j][i], predicted from logits[j]
                # We'll do a shift by i+1 for demonstration
                shift_idx = i + 1
                if shift_idx >= seq_len:
                    continue
                # Turn logits into [batch*seq_len, vocab_size]
                current_logits = logits[:, :-shift_idx, :].reshape(-1, vocab_size)
                current_labels = mtp_labels[:, shift_idx:, i].reshape(-1)
                token_loss = nn.functional.cross_entropy(current_logits, current_labels, ignore_index=-100)
                total_loss += token_loss
            loss = total_loss / self.mtp_tokens

        return logits, loss
