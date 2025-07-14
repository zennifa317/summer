import os

def create_versioned_dir(base_name="exp", dir="."):
    i = 1
    while True:
        dir_name = f"{base_name}_{i:02d}"
        target_path = os.path.join(dir, dir_name)

        if not os.path.exists(target_path):
            return target_path
        else:
            i += 1