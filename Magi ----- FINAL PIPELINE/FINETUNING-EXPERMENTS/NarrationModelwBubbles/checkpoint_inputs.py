"""Load independent model or layer checkpoints and cache frozen text features."""
from pathlib import Path
import copy
import shutil
import torch
from config import fingerprint, save_json
from bubble_features import model_files, sha256


def check_training_inputs(base_path, encoder_path, classifier_path, use_classifier,
                          encoder_layers, classifier_layers, saved=None):
    paths = {'DeBERTa base': base_path, 'DeBERTa checkpoint': encoder_path}
    if saved is None and use_classifier:
        paths['Classifier checkpoint'] = classifier_path
    if saved is None:
        paths.update({f'DeBERTa layer {index}': path for index, path in encoder_layers.items()})
        paths.update({f'Classifier module {name}': path for name, path in classifier_layers.items()})
    missing = [f'{name}: {path}' for name, path in paths.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError('Missing training inputs:\n' + '\n'.join(missing) +
                                '\nSelect an existing run/checkpoint in the first cell or config.py.')
    for name, path in paths.items():
        print(f'{name}: {path}')


def checkpoint_identity(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return model_files(path) if path.is_dir() else {'sha256': sha256(path)}


def read_weights(path, model_name):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if 'models' in saved:
        if model_name not in saved['models']:
            raise ValueError(f'{path} does not contain {model_name} weights.')
        return saved['models'][model_name], saved
    if 'state_dict' in saved:
        return saved['state_dict'], saved
    return saved, {}


def load_layer(module, path, prefix, model_name):
    weights, _ = read_weights(path, model_name)
    selected = {key[len(prefix) + 1:]: value for key, value in weights.items()
                if key.startswith(prefix + '.')}
    module.load_state_dict(selected or weights, strict=True)


def prepare_encoder(base_path, checkpoint, layer_checkpoints, cache_dir):
    from transformers import AutoModel, AutoTokenizer
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir() and not layer_checkpoints:
        return checkpoint
    identity = {'base': checkpoint_identity(base_path),
                'checkpoint': checkpoint_identity(checkpoint),
                'layers': {str(index): checkpoint_identity(path) for index, path in layer_checkpoints.items()}}
    folder = Path(cache_dir) / 'encoder_sources' / fingerprint(identity)
    if (folder / 'source.json').exists():
        return folder
    source = checkpoint if checkpoint.is_dir() else Path(base_path)
    encoder = AutoModel.from_pretrained(source, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    if checkpoint.is_file():
        weights, _ = read_weights(checkpoint, 'deberta')
        encoder.load_state_dict(weights, strict=True)
    for index, path in layer_checkpoints.items():
        if not isinstance(index, int) or not 0 <= index < len(encoder.encoder.layer):
            raise ValueError('DeBERTa layer keys must be zero-based integer layer indices.')
        load_layer(encoder.encoder.layer[index], path, f'encoder.layer.{index}', 'deberta')
    encoder.save_pretrained(folder, safe_serialization=True)
    tokenizer.save_pretrained(folder)
    save_json(folder / 'source.json', identity)
    return folder


def frozen_samples(data, encoder_path, cache_dir, max_tokens, chunk_size=2):
    from bubble_features import semantic_features
    signature = fingerprint({'source_samples': data['signature'], 'encoder': checkpoint_identity(encoder_path),
                             'max_tokens': max_tokens, 'context': 2, 'pool': 'target_mean_v1'})
    path = Path(cache_dir) / 'classifier_inputs' / signature / 'samples.pt'
    if path.exists():
        return torch.load(path, map_location='cpu', weights_only=True)
    result = copy.deepcopy(data)
    for sample in result['samples']:
        sequence_path = path.parent / 'sequences' / (sample['sequence_id'] + '.pt')
        if sequence_path.is_file():
            vectors = torch.load(sequence_path, map_location='cpu', weights_only=True)
        else:
            vectors = semantic_features(sample, encoder_path, max_tokens=max_tokens,
                                        batch_size=chunk_size, device='cuda' if torch.cuda.is_available() else 'cpu')
            sequence_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = sequence_path.with_suffix('.tmp')
            torch.save(vectors, temporary)
            temporary.replace(sequence_path)
        sample['text'] = torch.stack([vectors[box['box_id']] for box in sample['boxes']])
        print('Cached selected encoder:', sample['sequence_id'], flush=True)
    result['signature'] = signature
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    return result


def save_layer_files(run, models, metadata):
    run = Path(run)
    classifier = models['classifier']
    weights = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}
    torch.save(dict(metadata, state_dict=weights), run / 'classifier.pt')
    folder = run / 'classifier_layers'
    folder.mkdir(parents=True, exist_ok=True)
    for name, module in classifier.named_modules():
        if name and ('.' not in name or isinstance(module, (torch.nn.Linear, torch.nn.TransformerEncoderLayer))):
            torch.save({key: value.detach().cpu().clone() for key, value in module.state_dict().items()},
                       folder / f'{name}.pt')
    if 'deberta' in models:
        folder = run / 'deberta_layers'
        folder.mkdir(parents=True, exist_ok=True)
        for index, layer in enumerate(models['deberta'].encoder.layer):
            torch.save({key: value.detach().cpu().clone() for key, value in layer.state_dict().items()},
                       folder / f'layer_{index:02d}.pt')


def export_best(run, models, metadata, encoder_path, tokenizer=None):
    run = Path(run)
    if 'deberta' in models:
        encoder_path = run / 'deberta'
        models['deberta'].save_pretrained(encoder_path, safe_serialization=True)
        tokenizer.save_pretrained(encoder_path)
    else:
        encoder_path = Path(encoder_path)
        if model_files(encoder_path) != metadata['feature_contract']['deberta_files']:
            raise ValueError('Frozen encoder files changed during training; restore the selected checkpoint.')
        destination = run / 'deberta'
        if encoder_path.resolve() != destination.resolve():
            shutil.copytree(encoder_path, destination, dirs_exist_ok=True)
        encoder_path = destination
    contract = dict(metadata['feature_contract'], deberta_files=model_files(Path(encoder_path)))
    save_layer_files(run, models, dict(metadata, feature_contract=contract))
    best = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    best['feature_contract'] = contract
    temporary = run / 'best.pt.tmp'
    torch.save(best, temporary)
    temporary.replace(run / 'best.pt')
    source = ('from config import CONFIG as BASE_CONFIG, model_files\n'
              'CONFIG = dict(BASE_CONFIG)\n'
              f"CONFIG['deberta_model'] = {str(encoder_path)!r}\n")
    (run / 'inference_config.py').write_text(source, encoding='utf-8')
