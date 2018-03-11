import os
import shutil

from sklearn.model_selection import train_test_split


def train_val_test(ds, out_dir, ratio={'train': 0.6, 'test': 0.5}):
    train_dir = os.path.join(args.out_dir, 'train')
    os.mkdir(train_dir)
    val_dir = os.path.join(args.out_dir, 'val')
    os.mkdir(val_dir)
    test_dir = os.path.join(args.out_dir, 'test')
    os.mkdir(test_dir)
    train_f, vt_f, train_l, vt_l = train_test_split(ds.fpaths, ds.labels,
                                                    train_size=ratio['train'],
                                                    stratify=ds.labels)
    val_f, test_f, val_l, test_l = train_test_split(vt_f, vt_l,
                                                    train_size=ratio['test'],
                                                    stratify=vt_l)
    save_dataset(ds.cls_names, [train_dir, val_dir, test_dir],
                 [(train_f, train_l), (val_f, val_l), (test_f, test_l)])


def save_dataset(names, dir_lst, data_lst):
    for dir_path, data in zip(dir_lst, data_lst):
        for _, v in names.items():
            cls_dir = os.path.join(dir_path, v)
            os.mkdir(cls_dir)
        for fpath, label in zip(data[0], data[1]):
            shutil.copy(fpath, os.path.join(dir_path, names[label]))


if __name__ == '__main__':
    import argparse
    from process_dataset.proc_dataset import get_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('out_dir')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    ds = get_dataset(args.data_path)

    train_val_test(ds, args.out_dir)
