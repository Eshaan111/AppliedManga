# BestParamTuning

best.pt is the selected checkpoint; metadata.json records selection, scores, split and seeds. best_parameters.json contains full training and architecture settings. studies/ contains SQLite snapshots, trial tables and experiment manifests.

For inference, import BoxClassifier from the exported model.py. Instantiate with visual_dim and max_positions from best.pt, dropout from parameters, and (for the architecture model) num_heads and dim_feedforward from best_parameters.json. Load state_dict and call eval(). Pass crop_dim from the checkpoint. Inputs now also require dino: (B,N,384), extracted using crop_features.py and preprocessing.json with the same local frozen DINOv2. Use the same cached-feature preprocessing and frozen local encoder models as training.
