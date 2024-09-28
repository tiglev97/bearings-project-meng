import os
import zipfile
import tempfile

def ExtractZipFile(zip_file_path):
    try: 
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            temp_dir = tempfile.mkdtemp()
            zip_ref.extractall(temp_dir)
        return temp_dir
    except Exception as e:
        print(e)
