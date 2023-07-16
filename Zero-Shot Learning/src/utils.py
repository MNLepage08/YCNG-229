import pathlib

def clean_dir(cachedir):
    path_ = pathlib.Path( cachedir )
    if not path_.exists():
        return
    for p in pathlib.Path( cachedir ).iterdir():
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            clean_dir(p)
            p.rmdir()