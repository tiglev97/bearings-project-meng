import os
import zipfile
import tempfile

def extract_zip_file(zip_file_path):
    try: 
        temp_dir = tempfile.mkdtemp()

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        return temp_dir
    except Exception as e:
        print(e)
