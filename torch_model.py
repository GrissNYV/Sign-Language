import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """
    BiLSTM-based classifier equivalent to the prior Keras architecture:
      - 3 BiLSTM blocks (128, 128, 64) with dropout between layers
      - Dense(64) + BatchNorm + Dropout(0.5)
      - Dense(32) + Dropout(0.5)
      - Final classifier to n_classes
    Expects input of shape: (batch, seq_len, n_features)
    Returns logits of shape: (batch, n_classes)
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        lstm_hidden1: int = 128,
        lstm_hidden2: int = 128,
        lstm_hidden3: int = 64,
        dropout_lstm: float = 0.3,
        fc1_units: int = 64,
        fc2_units: int = 32,
        dropout_fc: float = 0.5,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes

        self.input_norm = nn.LayerNorm(n_features)

        self.lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden1,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout1 = nn.Dropout(dropout_lstm)

        self.lstm2 = nn.LSTM(
            input_size=lstm_hidden1 * 2,
            hidden_size=lstm_hidden2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout2 = nn.Dropout(dropout_lstm)

        self.lstm3 = nn.LSTM(
            input_size=lstm_hidden2 * 2,
            hidden_size=lstm_hidden3,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout3 = nn.Dropout(dropout_lstm)

        self.fc1 = nn.Linear(lstm_hidden3 * 2, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.relu = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(dropout_fc)

        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.dropout_fc2 = nn.Dropout(dropout_fc)

        self.classifier = nn.Linear(fc2_units, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = self.input_norm(x)

        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        out, _ = self.lstm2(out)
        out = self.dropout2(out)

        out, _ = self.lstm3(out)
        out = self.dropout3(out)

        # Take last timestep
        out = out[:, -1, :]  # (B, hidden3*2)

        out = self.fc1(out)
        out = self.relu(out)
        # BatchNorm1d expects (B, C)
        out = self.bn1(out)
        out = self.dropout_fc1(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout_fc2(out)

        logits = self.classifier(out)
        return logits


def load_bilstm_from_checkpoint(checkpoint_path: str, n_features: int, n_classes: int, device: torch.device) -> BiLSTMClassifier:
    model = BiLSTMClassifier(n_features=n_features, n_classes=n_classes)
    state = torch.load(checkpoint_path, map_location=device)
    # Allow loading state dict directly or wrapped with additional metadata
    state_dict = state.get('state_dict', state)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
