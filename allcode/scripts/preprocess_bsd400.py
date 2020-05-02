import argparse
import os
import os.path
import numpy as np
import h5py
import cv2
import glob
import tqdm


def main(args):
    print("Processing training data")
    scales = [1, 0.9, 0.8, 0.7]
    files = sorted(glob.glob(os.path.join(args.data_path, "Train400", "*.png")))
    with h5py.File(os.path.join(args.data_path, "train.h5"), "w") as h5f:
        train_size = 0
        for file in tqdm.tqdm(files):
            img = cv2.imread(file)
            h, w, c = img.shape
            for k in range(len(scales)):
                Img = cv2.resize(img, (int(h * scales[k]), int(w * scales[k])), interpolation=cv2.INTER_CUBIC)
                Img = np.expand_dims(Img[:, :, 0].copy(), 0) / 255.0
                patches = Im2Patch(Img, win=args.patch_size, stride=args.stride)
                for n in range(patches.shape[3]):
                    data = patches[:, :, :, n].copy()
                    h5f.create_dataset(str(train_size), data=data)
                    train_size += 1
                    for m in range(args.aug_times - 1):
                        data_aug = data_augmentation(data, np.random.randint(1, 8))
                        h5f.create_dataset(str(train_size) + "_aug_%d" % (m + 1), data=data_aug)
                        train_size += 1

    print("Processing validation data")
    files = sorted(glob.glob(os.path.join(args.data_path, "Set12", "*.png")))
    with h5py.File(os.path.join(args.data_path, "valid.h5"), "w") as h5f:
        valid_size = 0
        for file in tqdm.tqdm(files):
            img = cv2.imread(file)
            img = np.expand_dims(img[:, :, 0], 0) / 255.0
            h5f.create_dataset(str(valid_size), data=img)
            valid_size += 1

    print(f"Training size {train_size}, validation size {valid_size}")


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0 : endw - win + 0 + 1 : stride, 0 : endh - win + 0 + 1 : stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i : endw - win + i + 1 : stride, j : endh - win + j + 1 : stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def data_augmentation(image, mode):
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-path", default="data/bsd400", help="path to data directory")
    parser.add_argument("--patch-size", default=40, help="patch size")
    parser.add_argument("--stride", default=10, help="stride")
    parser.add_argument("--aug-times", default=1, help="number of augmentations")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
