"""Shared plain training loop for notebooks 07 and 08."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import fingerprint, save_json
from model import BoxClassifier, collate, metrics, prediction_rows, grad_norm


def split_manifest(data):
    return {
        'samples_signature': data['signature'],
        'train_sequences': [data['samples'][i]['sequence_id'] for i in data['train_indices']],
        'val_sequences': [data['samples'][i]['sequence_id'] for i in data['val_indices']],
        'train_counts': data['train_counts'].tolist(),
        'class_weights': data['class_weights'].tolist(),
    }


def validate_split(data):
    train = data['train_indices']
    val = data['val_indices']
    assert train and val and set(train).isdisjoint(val)
    assert len(set(train)) == len(train) and len(set(val)) == len(val)
    assert set(train) | set(val) == set(range(len(data['samples'])))
    train_ids = {data['samples'][i]['sequence_id'] for i in train}
    val_ids = {data['samples'][i]['sequence_id'] for i in val}
    assert train_ids.isdisjoint(val_ids)
    train_pages = {p['sha256'] for i in train for p in data['samples'][i]['pages']}
    val_pages = {p['sha256'] for i in val for p in data['samples'][i]['pages']}
    assert train_pages.isdisjoint(val_pages)
    for indices in [train, val]:
        labels = torch.cat([data['samples'][i]['labels'] for i in indices])
        assert (torch.bincount(labels[labels != -100], minlength=2) > 0).all()
    labels = torch.cat([data['samples'][i]['labels'] for i in train])
    assert torch.equal(torch.bincount(labels[labels != -100], minlength=2), data['train_counts'])


def run_epoch(model, loader, optimizer, criterion, device, training):
    model.train(training)
    loss_sum, count_sum, rows, norms = 0., 0, [], []
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        count = int((batch['labels'] != -100).sum())
        with torch.set_grad_enabled(training):
            logits = model(batch)
            summed = criterion(logits.reshape(-1, 2), batch['labels'].reshape(-1))
            if training and count:
                optimizer.zero_grad(set_to_none=True)
                (summed / count).backward()
                norms.append({'fusion': grad_norm(model.fusion),
                              'transformer': grad_norm(model.transformer),
                              'classifier': grad_norm(model.classifier)})
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        loss_sum += float(summed.detach())
        count_sum += count
        rows.extend(prediction_rows(batch, logits.detach().softmax(-1).cpu()))
    result = metrics(rows)
    result['loss'] = loss_sum / count_sum if count_sum else None
    if norms:
        result['grad_norm_mean_before_clip'] = {
            key: float(np.mean([x[key] for x in norms])) for key in norms[0]}
    return result, rows


def train_one_seed(data, params, seed, output_dir, reuse=True, device=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assert params['epochs'] > 0 and params['batch_size'] > 0
    assert params['lr'] > 0 and params['weight_decay'] >= 0
    assert 0 <= params['dropout'] < 1 and params['narration_sampling_weight'] > 0
    code = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ['model.py', 'training_runs.py']}
    run_id = fingerprint({'split': split_manifest(data), 'params': params, 'seed': seed, 'code': code,
                          'checkpoint_selection': 'macro_f1', 'version': 1})
    result_file = output_dir / 'result.json'
    checkpoint_file = output_dir / 'best.pt'
    if result_file.exists():
        result = json.loads(result_file.read_text(encoding='utf-8'))
        assert result['run_id'] == run_id, 'Run settings/data changed; use a new output folder.'
        if reuse and checkpoint_file.exists():
            print('Reusing completed run:', output_dir.name)
            return result

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    train = [data['samples'][i] for i in data['train_indices']]
    train = [s for s in train if (s['labels'] != -100).any()]
    val = [data['samples'][i] for i in data['val_indices']]
    sampling_weights = [params['narration_sampling_weight'] if (s['labels'] == 1).any() else 1.
                        for s in train]
    sampler = WeightedRandomSampler(sampling_weights, len(train), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train, batch_size=params['batch_size'], sampler=sampler, collate_fn=collate)
    val_loader = DataLoader(val, batch_size=params['batch_size'], shuffle=False, collate_fn=collate)
    visual_dim = train[0]['visual'].shape[-1]
    model = BoxClassifier(visual_dim, data['max_positions'], dropout=params['dropout']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = nn.CrossEntropyLoss(weight=data['class_weights'].to(device), ignore_index=-100, reduction='sum')
    history, best_score, best_result = [], -1., None
    save_json(output_dir / 'settings.json', {'run_id': run_id, 'seed': seed, 'params': params,
                                           'split': split_manifest(data)})
    for epoch in range(1, params['epochs'] + 1):
        train_metrics, _ = run_epoch(model, train_loader, optimizer, criterion, device, True)
        val_metrics, rows = run_epoch(model, val_loader, optimizer, criterion, device, False)
        assert np.isfinite(val_metrics['loss']) and np.isfinite(train_metrics['loss'])
        history.append({'epoch': epoch, 'train': train_metrics, 'val': val_metrics})
        save_json(output_dir / 'history.json', history)
        print(f"seed {seed} epoch {epoch:02d} | train {train_metrics['loss']:.4f} | "
              f"val {val_metrics['loss']:.4f} | narration F1 {val_metrics['NARRATED']['f1']:.3f} | "
              f"macro F1 {val_metrics['macro_f1']:.3f}")
        if val_metrics['macro_f1'] > best_score:
            best_score = val_metrics['macro_f1']
            best_result = {'run_id': run_id, 'seed': seed, 'best_epoch': epoch,
                'narration_precision': val_metrics['NARRATED']['precision'],
                'narration_recall': val_metrics['NARRATED']['recall'],
                'narration_f1': val_metrics['NARRATED']['f1'], 'macro_f1': best_score,
                'val_loss': val_metrics['loss'], 'metrics': val_metrics, 'params': params}
            torch.save({'state_dict': model.state_dict(), 'visual_dim': visual_dim,
                'max_positions': data['max_positions'], 'dropout': params['dropout'],
                'samples_signature': data['signature'], 'class_weights': data['class_weights'],
                'epoch': epoch, 'metrics': val_metrics, 'seed': seed, 'params': params,
                'run_id': run_id}, checkpoint_file)
            save_json(output_dir / 'best_validation_predictions.json', rows)
            pd.DataFrame(rows).to_csv(output_dir / 'validation_predictions.csv', index=False)
    # This file marks a fully completed seed, so interrupted seeds restart rather than look complete.
    save_json(result_file, best_result)
    del model, optimizer, criterion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_result


def summarize_runs(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = ['narration_precision', 'narration_recall', 'narration_f1', 'macro_f1']
    per_seed = pd.DataFrame([{k: r[k] for k in ['seed', 'best_epoch', 'val_loss'] + keys}
                             for r in results])
    assert len(per_seed) and not per_seed.seed.duplicated().any()
    summary = per_seed[keys].agg(['mean', 'std', 'min', 'max']).T
    summary['range'] = summary['max'] - summary['min']
    per_seed.to_csv(output_dir / 'per_seed_metrics.csv', index=False)
    summary.to_csv(output_dir / 'seed_summary.csv', index_label='metric')
    return per_seed, summary
