import argparse
import sys

import zarr


def parse_args():
    parser = argparse.ArgumentParser(description='copy zarr (DirectoryStore) to (ZipStore)')
    parser.add_argument('--zarr-path', help='output directory', required=True, type=str)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    path = args.zarr_path

    # consolidate metadata
    store = zarr.DirectoryStore(path)
    zarr.consolidate_metadata(store)

    # copy to zip
    path2 = path.rstrip('.zarr') + '.zip'
    store2 = zarr.ZipStore(path2, mode='w')
    store1 = zarr.DirectoryStore(path)
    zarr.copy_store(store1, store2, log=sys.stdout)
    store2.close()
