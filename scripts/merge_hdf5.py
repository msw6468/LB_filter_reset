import os, sys
import argparse
import colorful
from pprint import pprint
from tqdm import tqdm

from glob import glob
import h5py

LOAD_DIR = {
    'ai2'    : '/net/nfs3.prior/dongjook/',
    'tate'   : '/gallery_tate/dongyeon.woo/howto100m/',
    'orsay'  : '/gallery_orsay/sangwoo.moon/data/video/howto100m/',
    'moma'   : '/gallery_moma/sangwoo.moon/data/video/howto100m/',
    'millet' : '/gallery_millet/chris.kim/data/howto100m/',
    'getty'  : '/gallery_getty/dongjoo.kim/vision/howto370k/',}

META_PART = {
    'tate' : 0,
    'moma' : 1,
    'orsay': 2,
    'getty': 3,
    'ai2'  : 4,}

DATA_TYPE = [
    # For Ours
    # 'preprocessed_frames',
    # 'frame_emb',
    # 'real_text_emb',
    # 'real_text_sim',
    # 'synt_text_emb',
    # 'synt_text',
    # 'synt_text_sim',

    # For LB
    'text_ids',
    'text_emb',
    'clip_emb',
    'clip_sim',
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_name",     type=str, required=True,   help="[moma, tate, getty, orsay, ai2]")
    parser.add_argument("--data_version", type=str, required=True,   help="[subset, 730k, 370k]")
    parser.add_argument("--merge_type",   type=str, required=True,   help="[merge_meta_part, merge_all]")
    parser.add_argument("--meta_part",    type=int,                  help="[0, 1, 2, ...]")
    parser.add_argument("--debug",        type=str, default="False", help="[True, False]")
    return parser.parse_args()


def main(args):
    args.debug     = True if args.debug     == "True" else False

    if args.dir_name == 'ai2':
        if args.merge_type == 'merge_all':
            """ merge all files based on data_version, data_type
            """
            root_dir = os.path.join(LOAD_DIR['ai2'], args.data_version)
            for data_type in DATA_TYPE:
                target_file = os.path.join(root_dir, f'{data_type}_total.h5')
                target_h5  = h5py.File(target_file, 'a')
                os.chmod(target_h5.filename, mode=0o777)
                if data_type == 'preprocessed_frames':
                    source_file_list = glob(os.path.join(root_dir, f'{data_type}_part*.h5py'))
                else:
                    source_file_list = glob(os.path.join(root_dir, f'{data_type}_part{args.meta_part}_*_*.h5'))

                source_file_list.sort()
                count = 0
                for source_file in tqdm(source_file_list):
                    source_h5 = h5py.File(source_file, 'r')
                    for vid in tqdm(source_h5.keys()):
                        target_h5.create_dataset(vid, data = source_h5[vid][...])
                print(count)
                target_h5.flush()
                target_h5.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print(colorful.bold_pink("Thank you and Good Job Computer.").styled_string)
