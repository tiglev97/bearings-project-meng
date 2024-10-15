import os
from pipelines.DataConverter import extract_folder_path
from pipelines.Excel2Jsonl import excel_to_jsonl
from pipelines.ZipExtractor import extract_zip_file



def data_entry(input):

    base_path= extract_zip_file(input)
    output_path = "outputs"

    folder_path , jsonl_file_path = extract_folder_path(base_path,output_path)


    for i in range(len(folder_path)):
        excel_to_jsonl(folder_path[i], jsonl_file_path[i])
    

