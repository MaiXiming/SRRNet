# import numpy as np
from statistics import mean, stdev
import csv
import os

# current_path = os.getcwd()
path_file = os.path.abspath(__file__)
path_folder = os.path.dirname(path_file)

# subjects = 35
source_path = os.path.join(path_folder, 'Results/CSV/')
destination_path = os.path.join(path_folder, 'Results/Output/')
# source_path = os.path.join(current_path, '../Save/PLTSCompare/ucsd/SSVEPNet-NSCC/')
# destination_path = os.path.join(current_path, '../Save/PLTSCompare/ucsd/SSVEPNet-NSCC/')

def main():
    # source_path = os.path.join(current_path, 'Results/PLTSCompare/')
    csv_files = get_csv_filenames(source_path)
    for file in csv_files:
        data = get_csv_content(os.path.join(source_path, file))
        dataset = file[0:4]
        if dataset == 'benc':
            subjects = 35
        elif dataset == 'beta':
            subjects = 70
        else:
            raise ValueError("dataset not found!")
        
        update_csv_content(file, data, destination_path, subjects)

def get_csv_content(csv_path):
    with open(csv_path, 'r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)
        idx = 0
    # return data[idx:(idx+subjects)]
    return data


def update_csv_content(filename, data, output_folder, subjects):
    accs_test, loss_train, loss_valid, loss_unseen, corr_train, corr_valid, corr_unseen = [],[],[],[],[],[],[]
    # subjects = len(data) - 4
    for subject in range(subjects):
        accs_test.append(float(data[subject][0]))
        loss_train.append(float(data[subject][1]))
        loss_valid.append(float(data[subject][2]))
        loss_unseen.append(float(data[subject][3]))
        corr_train.append(float(data[subject][4]))
        corr_valid.append(float(data[subject][5]))
        corr_unseen.append(float(data[subject][6]))

    # # data.insert(0, ['test', 'val'])
    # data.append(['test', 'val'])
    # data.append(['mean'])
    # # data.append([np.mean(accs_test), np.mean(accs_val)])
    # data.append([mean(accs_test), mean(loss_train), mean(loss_valid), mean(loss_unseen), mean(corr_train), mean(corr_valid), mean(corr_unseen)])
    # data.append(['std'])
    # # data.append([np.std(accs_test), np.std(accs_val)])
    # data.append([stdev(accs_test), stdev(loss_train), stdev(loss_valid), stdev(loss_unseen), stdev(corr_train), stdev(corr_valid), stdev(corr_unseen)])

    # data.insert(0, ['test', 'val'])
    insert_idx = subjects
    data.insert(insert_idx, ['test', 'val'])
    insert_idx += 1
    data.insert(insert_idx, ['mean'])
    insert_idx += 1
    # data.append([np.mean(accs_test), np.mean(accs_val)])
    data.insert(insert_idx, [mean(accs_test), mean(loss_train), mean(loss_valid), mean(loss_unseen), mean(corr_train), mean(corr_valid), mean(corr_unseen)])
    insert_idx += 1
    data.insert(insert_idx, ['std'])
    insert_idx += 1
    # data.append([np.std(accs_test), np.std(accs_val)])
    data.insert(insert_idx, [stdev(accs_test), stdev(loss_train), stdev(loss_valid), stdev(loss_unseen), stdev(corr_train), stdev(corr_valid), stdev(corr_unseen)])
    insert_idx += 1

    # output_folder = os.path.join(current_path, 'Output/')
    # output_folder = path
    if os.path.exists(output_folder):
            pass
    else:
        os.makedirs(output_folder)

    csv_path = os.path.join(output_folder, filename)
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


def get_csv_filenames(folder_path):
    # 获取文件夹中的所有文件和目录名
    entries = os.listdir(folder_path)

    # 筛选出所有以.csv结尾的文件
    csv_files = [file for file in entries if file.endswith('.csv')]

    return csv_files


if __name__ == '__main__':
    main()
# source_path = os.path.join(current_path, 'PLTSCompare/')
# filename = 'ucsd-SSVEPNet-0.5-inter-normal-0.csv'
# csv_path = os.path.join(source_path, filename)

# data = get_csv_content(csv_path)
# update_csv_content(filename, data)
