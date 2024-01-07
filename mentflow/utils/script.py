import os
import pathlib
import pickle
import shutil
import sys
import time

from .utils import save_pickle


class ScriptManager:
    """Helps setup paths, logger, etc.

    Parameters
    ----------
    file_path : str
        Full path to the script file.
    output_dir : str
        Full path to output directory. The directory will be created if it does 
        not exist. Files will be saved to /{output_dir}/{file_path.stem}/{timestamp}. 
        Example:
            - file_path = "/path/to/script_name.py"
            - output_dir = "/path/to/output/"
            - timestamp = "240104215723"
            - output_dir --> "/path/to/output/script_name/240104215723/"
    """
    def __init__(self, file_path: str, output_dir : str) -> None:
        self.datestamp = time.strftime("%Y-%m-%d")
        self.timestamp = time.strftime("%y%m%d%H%M%S")
        self.file_path = pathlib.Path(file_path)
        self.output_dir = os.path.join(output_dir, self.file_path.stem, self.timestamp)
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

    def get_path(self, filename : str) -> str:
        """Get full path to output file name."""
        return os.path.join(self.output_dir, filename)

    def make_dirs(self, *dir_names) -> None:
        """Make directories in output folder."""
        for dir_name in dir_names:
            path = os.path.join(self.output_dir, dir_name)
            if not os.path.exists(path):
                os.makedirs(path)

    def save_pickle(self, object, filename):
        """Pickle object and save to output directory."""
        return save_pickle(object, self.get_path(filename))

    def save_script_copy(self):
        """Save a copy of the script file to the output directory."""
        old_path = str(self.file_path)
        new_path = self.get_path(self.file_path.name)
        shutil.copy(old_path, new_path)
