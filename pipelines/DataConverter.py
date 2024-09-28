import os


def ExtractFolderPath(base_path,output_path):
    folder_path = []
    jsonl_file_path=[]
    for folder in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, folder)):
            folder_path .append(os.path.join(base_path, folder))
            jsonl_file_path.append(os.path.join(output_path,folder+".jsonl"))
    return folder_path , jsonl_file_path


