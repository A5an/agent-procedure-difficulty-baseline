"""Fine-tuned bge-small regressor on fold-IRT difficulty (copied from prototype/baseline/extra_methods.py, unchanged)."""
import numpy as np
import torch
from scipy.special import expit


class FineTunedEncoder:
    """Fine-tuned text regressor: an encoder (bge-base) with a linear head is trained on the
    training tasks' statements to predict their IRT difficulty (fine-tuned assessor family:
    item-difficulty regression with fine-tuned transformers (bge-small, one seed per split to fit
    the night), the strongest text baseline in the
    education shared tasks and in ADeLe's comparison). P = sigmoid(theta_agent - beta_hat)."""

    def __init__(self, texts, name="finetuned", base="BAAI/bge-small-en-v1.5", epochs=4, lr=5e-5,
                 batch=16, max_len=512, seed=0):
        self.texts, self.name, self.base = texts, name, base
        self.epochs, self.lr, self.batch, self.max_len, self.seed = epochs, lr, batch, max_len, seed
        self._predicted_difficulties = {}
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

    def _encode(self, model, tok, ids):
        enc = tok([self.texts[i] for i in ids], padding=True, truncation=True,
                  max_length=self.max_len, return_tensors="pt").to(self.device)
        out = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        return (out * m).sum(1) / m.sum(1)

    def fit(self, data, train_task_ids):
        from transformers import AutoModel, AutoTokenizer
        torch.manual_seed(self.seed)
        self._predicted_difficulties = {}
        self.tok = AutoTokenizer.from_pretrained(self.base)
        self.model = AutoModel.from_pretrained(self.base).to(self.device)
        self.head = torch.nn.Linear(self.model.config.hidden_size, 1).to(self.device)
        train = list(train_task_ids)
        y = data.train_items.loc[train, "b"].values.astype(np.float32)
        self.mu, self.sd = float(y.mean()), float(y.std() + 1e-6)
        yt = torch.tensor((y - self.mu) / self.sd, device=self.device)
        opt = torch.optim.AdamW(list(self.model.parameters()) + list(self.head.parameters()), lr=self.lr,
                                weight_decay=0.01)
        rng = np.random.default_rng(self.seed)
        self.model.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(train))
            for s in range(0, len(train), self.batch):
                idx = order[s:s + self.batch]
                pred = self.head(self._encode(self.model, self.tok, [train[i] for i in idx])).squeeze(-1)
                loss = torch.nn.functional.mse_loss(pred, yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.model.eval()

    def predict_probability(self, data, agent_id, task_id):
        if not self._predicted_difficulties:
            test = list(data.test_tasks)
            with torch.no_grad():
                preds = []
                for s in range(0, len(test), 32):
                    z = self.head(self._encode(self.model, self.tok, test[s:s + 32])).squeeze(-1)
                    preds += (z.float().cpu().numpy() * self.sd + self.mu).tolist()
            self._predicted_difficulties = dict(zip(test, preds))
            # free memory before the next fold
            del self.model
            if self.device == "mps":
                torch.mps.empty_cache()
        theta = data.train_abilities.loc[agent_id, "ability"]
        return float(expit(theta - self._predicted_difficulties[task_id]))
