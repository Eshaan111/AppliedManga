import numpy as np
import torch
from torch import nn
from sklearn.metrics import precision_recall_fscore_support

class BoxClassifier(nn.Module):
    def __init__(self, visual_dim, max_positions, dropout=0.1, num_heads=4, dim_feedforward=512):
        super().__init__()
        assert num_heads > 0 and 256 % num_heads == 0, 'num_heads must divide model width 256.'
        assert dim_feedforward > 0
        self.text_projection = nn.Linear(768, 256)
        self.visual_projection = nn.Linear(visual_dim, 256)
        self.geometry_projection = nn.Linear(4, 32)
        self.tail_projection = nn.Linear(2, 16)
        self.fusion = nn.Sequential(nn.Linear(560, 256), nn.ReLU(), nn.Dropout(dropout))
        self.position = nn.Embedding(max_positions, 256)
        self.page_embedding = nn.Embedding(3, 256)
        layer = nn.TransformerEncoderLayer(d_model=256, nhead=num_heads, dim_feedforward=dim_feedforward,
                                           dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.classifier = nn.Linear(256, 2)

    def forward(self, batch):
        parts = [torch.relu(self.text_projection(batch['text'])),
                 torch.relu(self.visual_projection(batch['visual'])),
                 torch.relu(self.geometry_projection(batch['geometry'])),
                 torch.relu(self.tail_projection(batch['tail']))]
        tokens = self.fusion(torch.cat(parts, dim=-1))
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        tokens = tokens + self.position(positions)[None] + self.page_embedding(batch['page'])
        tokens = self.transformer(tokens, src_key_padding_mask=~batch['attention_mask'])
        return self.classifier(tokens)

def collate(samples):
    n = max(len(s['boxes']) for s in samples)
    batch = {}
    for key in ['text', 'visual', 'geometry', 'tail', 'page', 'labels']:
        first = samples[0][key]
        shape = (len(samples), n) + first.shape[1:]
        batch[key] = torch.full(shape, -100 if key == 'labels' else 0, dtype=first.dtype)
        for i, sample in enumerate(samples):
            batch[key][i, :len(sample['boxes'])] = sample[key]
    batch['attention_mask'] = torch.arange(n)[None] < torch.tensor([len(s['boxes']) for s in samples])[:, None]
    batch['samples'] = samples
    return batch

def metrics(rows):
    labeled = [r for r in rows if r['label'] != -100]
    assert labeled, 'No supervised boxes'
    y = np.array([r['label'] for r in labeled])
    pred = np.array([r['prediction'] for r in labeled])
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    result = {'macro_f1': float(f1.mean()), 'balanced_accuracy': float(recall[support > 0].mean())}
    for c, name in enumerate(['SPOKEN', 'NARRATED']):
        probabilities = np.array([r['p_spoken'] if c == 0 else r['p_narrated'] for r in labeled if r['label'] == c])
        result[name] = {'precision': float(precision[c]), 'recall': float(recall[c]), 'f1': float(f1[c]), 'support': int(support[c]),
                        'p_true_mean': float(probabilities.mean()) if len(probabilities) else None,
                        'p_true_median': float(np.median(probabilities)) if len(probabilities) else None,
                        'p_true_gt_0.9': float((probabilities > .9).mean()) if len(probabilities) else None,
                        'p_true_lt_0.5': float((probabilities < .5).mean()) if len(probabilities) else None}
    return result

def prediction_rows(batch, probabilities):
    rows = []
    for i, sample in enumerate(batch['samples']):
        for j, box in enumerate(sample['boxes']):
            p = probabilities[i, j].tolist()
            rows.append({'sequence_id': sample['sequence_id'], 'box_id': box['box_id'], 'ocr': box['ocr'],
                'page_index': box['page_index'], 'bbox': box['bbox'], 'label': box['label'],
                'gt': {0: 'SPOKEN', 1: 'NARRATED', -100: 'IGNORED'}[box['label']],
                'prediction': int(p[1] > p[0]), 'predicted_class': 'NARRATED' if p[1] > p[0] else 'SPOKEN',
                'p_spoken': p[0], 'p_narrated': p[1], 'tail_score': float(sample['tail'][j, 0]),
                'has_tail': int(sample['tail'][j, 1])})
    return rows

def grad_norm(module):
    return sum(float(p.grad.detach().square().sum()) for p in module.parameters() if p.grad is not None) ** .5
