"""Small checkpoint helpers shared by the two training notebooks."""
from pathlib import Path
import random
import numpy as np
import torch
from config import save_json


def load_resume(run, resume):
    path = Path(run) / 'last.pt'
    if path.exists():
        if not resume:
            raise FileExistsError('Use ResumeTraining=True or choose a new RUN folder.')
        return torch.load(path, map_location='cpu', weights_only=True)
    if Path(run).exists() and any(Path(run).iterdir()):
        raise FileExistsError('This folder has no resumable last.pt; choose a new RUN and set initialization paths.')
    return None


def restore_state(saved, models, optimizer, scaler, sampler, contract):
    if saved is None:
        return
    if saved['training_contract'] != contract:
        raise ValueError('Inputs or training setup changed; choose a new RUN for this experiment.')
    for name, model in models.items():
        model.load_state_dict(saved['models'][name], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(saved['optimizer'])
    if scaler is not None:
        scaler.load_state_dict(saved['scaler'])
    random.setstate(saved['rng']['python'])
    name, keys, pos, gaussian, cached = saved['rng']['numpy']
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, gaussian, cached))
    torch.set_rng_state(saved['rng']['torch'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved['rng']['cuda'])
    sampler.generator.set_state(saved['rng']['sampler'])


def save_progress(run, models, optimizer, scaler, sampler, metadata,
                  epoch, history, best_f1, best_epoch, metrics, improved):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    numpy_state = np.random.get_state()
    states = {name: {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
              for name, model in models.items()}
    saved = dict(metadata, models=states, state_dict=states['classifier'], epoch=epoch,
                 history=history, best_f1=best_f1, best_epoch=best_epoch, metrics=metrics,
                 optimizer=optimizer.state_dict() if optimizer is not None else None,
                 scaler=scaler.state_dict() if scaler is not None else None,
                 rng={'python': random.getstate(),
                      'numpy': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                      'torch': torch.get_rng_state(),
                      'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                      'sampler': sampler.generator.get_state()})
    for name in ['last.pt'] + (['best.pt'] if improved else []):
        temporary = run / (name + '.tmp')
        torch.save(saved, temporary)
        temporary.replace(run / name)
    save_json(run / 'history.json', history)
