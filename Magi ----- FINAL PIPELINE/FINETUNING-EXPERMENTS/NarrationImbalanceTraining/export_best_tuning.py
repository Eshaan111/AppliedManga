"""Export the best completed Optuna candidate without training or changing a study."""
import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path


def export_best_tuning(root):
    import optuna
    import torch

    root = Path(root)
    candidates = []
    studies = []
    for folder_name, study_name in [
        ('optuna_tuning', 'fixed_split_macro_f1_v1'),
        ('optuna_architecture', 'fixed_split_architecture_v1'),
        ('optuna_joint', 'fixed_split_joint_refinement_v1'),
    ]:
        folder = root / 'runs' / folder_name
        database = folder / 'study.db'
        manifest_file = folder / 'manifest.json'
        if not database.exists() or not manifest_file.exists():
            continue
        manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
        study = optuna.load_study(study_name=study_name,
            storage='sqlite:///' + database.resolve().as_posix())
        completed = [trial for trial in study.trials
                     if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None]
        if not completed:
            continue
        studies.append((folder, study, manifest))
        for trial in completed:
            candidates.append((float(trial.value), folder, study, manifest, trial))

    if not candidates:
        print('No completed tuning trials available. Run the search first, then rerun this cell.')
        return

    # Cross-study comparison is valid only for the identical split and search seeds.
    reference = studies[0][2]
    for _, _, manifest in studies:
        assert manifest['split'] == reference['split'], 'Study splits differ; cannot select an overall winner.'
        assert manifest['search_seeds'] == reference['search_seeds'], 'Study search seeds differ.'
    score, folder, study, manifest, trial = max(candidates, key=lambda item: item[0])
    params = dict(manifest['base_params'])
    params.update(trial.params)
    params.setdefault('num_heads', 4)
    params.setdefault('dim_feedforward', 512)

    # Prefer the five-seed report when available; otherwise use completed search-seed runs.
    runs = {}
    for run_folder in [folder / f'trial_{trial.number:03d}',
                       folder / f'best_trial_{trial.number:03d}_five_seeds']:
        for result_file in sorted(run_folder.glob('seed_*/result.json')):
            result = json.loads(result_file.read_text(encoding='utf-8'))
            result_params = dict(result['params'])
            result_params.setdefault('num_heads', 4)
            result_params.setdefault('dim_feedforward', 512)
            checkpoint = result_file.parent / 'best.pt'
            if result_params == params and checkpoint.exists():
                runs[result['seed']] = (result, checkpoint)
    assert runs, 'Winning trial has no complete checkpoint artifacts; restore its run folders first.'
    result, checkpoint_file = max(runs.values(), key=lambda item: item[0]['macro_f1'])
    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
    assert checkpoint['run_id'] == result['run_id'], 'Checkpoint and seed result do not match.'
    assert checkpoint['samples_signature'] == manifest['split']['samples_signature']

    output = root / 'BestParamTuning'
    output.mkdir(exist_ok=True)
    shutil.copy2(checkpoint_file, output / 'best.pt')
    architecture = folder.name != 'optuna_tuning'
    model_file = 'model_architecture.py' if architecture else 'model.py'
    shutil.copy2(root / model_file, output / 'model.py')
    metadata = {
        'winning_study': study.study_name, 'trial_number': trial.number,
        'trial_mean_macro_f1': score, 'parameters': params,
        'checkpoint_seed': result['seed'], 'checkpoint_best_epoch': result['best_epoch'],
        'checkpoint_macro_f1': result['macro_f1'], 'checkpoint_metrics': result['metrics'],
        'source_checkpoint': str(checkpoint_file), 'source_model_file': model_file,
        'selection': 'Highest mean macro F1 across fixed search seeds; then highest validation macro F1 seed within that trial.',
        'available_checkpoint_seeds': sorted(runs),
        'split': manifest['split'], 'search_seeds': manifest['search_seeds'],
        'report_seeds': manifest['report_seeds'], 'study_user_attrs': study.user_attrs,
        'caveat': 'Selection uses development validation data; this is not an independent test score.',
        'compared_studies': [{'study': s.study_name, 'best_mean_macro_f1': s.best_value,
                             'completed_trials': sum(t.state == optuna.trial.TrialState.COMPLETE for t in s.trials)}
                            for _, s, _ in studies],
    }
    for name, value in [('metadata.json', metadata), ('best_parameters.json', params)]:
        (output / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    # SQLite backup captures committed trials safely even while a notebook has a connection open.
    study_output = output / 'studies'
    study_output.mkdir(exist_ok=True)
    for study_folder, saved_study, saved_manifest in studies:
        prefix = study_folder.name
        with closing(sqlite3.connect(str(study_folder / 'study.db'))) as source:
            with closing(sqlite3.connect(str(study_output / f'{prefix}.db'))) as destination:
                source.backup(destination)
        saved_study.trials_dataframe().to_csv(study_output / f'{prefix}_trials.csv', index=False)
        (study_output / f'{prefix}_manifest.json').write_text(
            json.dumps(saved_manifest, indent=2) + '\n', encoding='utf-8')
    (output / 'README.md').write_text(
        '# BestParamTuning\n\n'
        'best.pt is the selected checkpoint; metadata.json records selection, scores, split and seeds. '
        'best_parameters.json contains full training and architecture settings. '
        'studies/ contains SQLite snapshots, trial tables and experiment manifests.\n\n'
        'For inference, import BoxClassifier from the exported model.py. Instantiate with '
        'visual_dim and max_positions from best.pt, dropout from parameters, and (for the architecture model) '
        'num_heads and dim_feedforward from best_parameters.json. Load state_dict and call eval(). '
        'Use the same cached-feature preprocessing and frozen local encoder models as training.\n', encoding='utf-8')
    print('Exported:', output)
    print('Winning study:', study.study_name, '| trial:', trial.number, '| mean macro F1:', score)
    print('Checkpoint seed:', result['seed'], '| validation macro F1:', result['macro_f1'])
    return output
