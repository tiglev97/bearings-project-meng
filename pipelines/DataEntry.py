from pipelines.DataConverter import ExtractFolderPath
from pipelines.Excel2Jsonl import excel_to_jsonl
from pipelines.ZipExtractor import ExtractZipFile



def data_entry(input):

    base_path= ExtractZipFile(input)
    output_path="outputs"
    folder_path , jsonl_file_path = ExtractFolderPath(base_path,output_path)


    for i in range(len(folder_path)):
        excel_to_jsonl(folder_path[i], jsonl_file_path[i])


if __name__ == '__main__':
    data_entry()
