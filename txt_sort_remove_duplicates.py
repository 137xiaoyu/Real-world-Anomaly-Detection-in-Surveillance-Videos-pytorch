import os


if __name__ == '__main__':
    old_files = ['train_anomaly.txt', 'train_normal.txt', 'test_anomalyv2.txt', 'test_normalv2.txt']
    for old_file in old_files:
        with open(old_file, 'r') as f:
            contents = sorted(list(set(f.readlines())))
        with open(os.path.splitext(old_file)[0] + '_sorted.txt', 'w') as new_f:
            new_f.writelines(contents)
