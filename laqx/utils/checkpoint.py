import os


def find_last_checkpoint(ckpt_path):
    if ckpt_path and os.path.exists(ckpt_path):
        files = [f for f in os.listdir(ckpt_path) if 'ckpt_' in f]
        for file in sorted(files, reverse=True):
            fname = os.path.join(ckpt_path, file)
            return fname
    return None
