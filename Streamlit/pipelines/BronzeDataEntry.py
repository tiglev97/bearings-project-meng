import os

def get_bronze_data_path():
    #open the folder and read the contents from C:\uoft\Meng_project\bearings-project-meng\Streamlit\outputs
    path = 'outputs\\Bronze\\'
    files = os.listdir(path)
    print(files)
    path_list=[]

    for file in files:
        if file.endswith('.jsonl'):
            path_list.append(path+file)

    return path_list

