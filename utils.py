import pathlib
import shutil
import csv

def delete_dir(path):
    shutil.rmtree(path)
    #ファイル一個ずつ削除する
    # check_dir = pathlib.Path(path)
    # for file in check_dir.iterdir():
    #     if file.is_file():
    #         file.unlink

def make_dir(path):
    delete_dir(path)
    pathlib.Path(path).mkdir()

def Save2Csv(save_data, header, save_path : str, save_mode = 'w'):
    # 保存するデータの格納形式を判定
    if isinstance(save_data, list) and isinstance(save_data[0], list):
        is2D = True
    elif isinstance(save_data, list) and not isinstance(save_data[0], list):
        is2D = False
    else:
        return -1
    if is2D:
        if save_mode == "w":
            with open(save_path, save_mode, encoding="utf-8", errors="replace") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for i, data in enumerate(save_data):
                    writer.writerow(data)
        # elif save_mode == "a":
        #     with open(save_path, save_mode, encoding="utf-8", errors="replace") as f:
        #         writer = csv.writer(f)
        #         writer.writerow("\n")
        #         writer.writerow(header)
        #         for i, row in zip(index, save_data):
        #             writer.writerow([i] + row)
    else:
        writer.writerow(save_data)

    return 0