from pathlib import Path 
from sklearn.model_selection import train_test_split


def get_seq_dirs(dir_path):
    dir = Path(dir_path)
    seqs = [item.name for item in dir.iterdir()]
    return list(seqs)


def split_seqs(seq_list: list, v_size, random_state=42):
    """Split sequence IDs reproducibly."""
    train_seqs, val_seqs = train_test_split(
        seq_list,
        test_size=v_size,
        random_state=random_state,
    )
    
    return train_seqs,val_seqs

def fit_img(dir_path, v_size, random_state=42):
    seqs = get_seq_dirs(dir_path)
    X, y = split_seqs(seqs, v_size, random_state=random_state)
    
    return X,y

def fit_seq(seq_list, v_size, random_state=42):
    X, y = split_seqs(seq_list, v_size, random_state=random_state)
    
    return X, y
