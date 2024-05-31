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
    'frame_emb',
    'real_text_emb',
    'real_text_sim',
    # 'synt_text_emb',
    # 'synt_text',
    # 'synt_text_sim',

    # For LB
    # 'clip_sim',
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_name",     type=str, required=True,   help="[moma, tate, getty, orsay, ai2]")
    parser.add_argument("--merge_type",   type=str, required=True,   help="[merge_meta_part, merge_all]")
    parser.add_argument("--data_version", type=str, required=True,   help="[subset, 730k, 370k]")
    parser.add_argument("--meta_part",    type=int,                  help="[0, 1, 2, ...]")
    parser.add_argument("--exclude_0",    type=str, default="False", help="[True, False]")
    parser.add_argument("--debug",        type=str, default="False", help="[True, False]")

    return parser.parse_args()


def main(args):
    args.exclude_0 = True if args.exclude_0 == "True" else False
    args.debug     = True if args.debug     == "True" else False

    if args.dir_name == 'ai2':
        if args.merge_type == 'merge_all':
            """ merge all files based on data_version, data_type
            """
            root_dir         = os.path.join(LOAD_DIR['ai2'], args.data_version)
            for data_type in DATA_TYPE:
                target_file = os.path.join(root_dir, f'{data_type}_total.h5')
                target_f    = h5py.File(target_file, 'a')
                os.chmod(target_f.filename, mode=0o777)
                if data_type == 'preprocessed_frames':
                    source_file_list = glob(os.path.join(root_dir, f'{data_type}_part*.h5py'))
                else:
                    source_file_list = glob(os.path.join(root_dir, f'{data_type}_part*_*_*.hdf5'))
                source_file_list.sort()
                count = 0
                for source_file in tqdm(source_file_list):
                    if args.debug:
                        source_f = h5py.File(source_file, 'r')
                        source_f_count = len(source_f.keys())
                        count += source_f_count
                        print(source_f, source_f_count)
                    else:
                        source_f = h5py.File(source_file, 'r')
                        for vid in source_f.keys():
                            target_f.create_dataset(vid, data = source_f[vid][...])
                print(count)
                target_f.flush()
                target_f.close()




    # if args.dir_name == 'merge_htm100m':
    #     for data_type in DATA_TYPE:
    #         root_dir         = os.path.join(LOAD_DIR['millet'])
    #         target_f         = h5py.File(os.path.join(root_dir, 'total', f'{data_type}_total.h5'), 'a')
    #         # source_file_list = glob(os.path.join(root_dir, '*', f'{data_type}.h5'))
    #         source_file_list = ['/gallery_millet/chris.kim/data/howto100m/370k/real_text_sim_part1_8_6.hdf5']
    #         for source_file in source_file_list:
    #             if '730k' in source_file:
    #                 continue
    #             # if '370k' in source_file:
    #             #     continue
    #             if 'subset' in source_file:
    #                 continue
    #             source_f = h5py.File(os.path.join(source_file), 'r')
    #             print(f'merge from {source_f.filename} to {target_f.filename}')
    #             for key in tqdm(source_f.keys()):
    #                 if key in target_f.keys():
    #                     print(key)
    #                 else:
    #                     target_f.create_dataset(key, data = source_f[key][...])

    #         target_f.flush()
    #         target_f.close()


    # elif args.dir_name in ['moma', 'tate', 'orsay', 'getty']:
    #     root_dir = os.path.join(LOAD_DIR['millet'], args.data_version)
    #     for data_type in DATA_TYPE:
    #         print(data_type)
    #         merged_f = h5py.File(os.path.join('/gallery_moma/sangwoo.moon/data/video/howto100m/730k', f'{data_type}_total.h5'), 'a')
    #         file_list = glob(os.path.join(root_dir, f'{data_type}_part{args.meta_part}_*.hdf5'))
    #         file_list.sort()
    #         for file_path in tqdm(file_list):
    #             if (args.exclude_0 == True) and ('_0' in file_path):
    #                 continue
    #             cur_f = h5py.File(file_path, 'r')
    #             for key in tqdm(cur_f.keys()):
    #                 merged_f.create_dataset(key, data = cur_f[key][...])

    #         merged_f.flush()
    #         merged_f.close()
    # else:
    #     if args.data_version == '730k':
    #         if 'final_merge' in args.dir_name:
    #             for data_type in DATA_TYPE:
    #                 # ai2 server datasize > our server dataset
    #                 # target: ai2.h5
    #                 # source: ours.h5
    #                 target_f = h5py.File(os.path.join('/gallery_millet/chris.kim/data/howto100m', args.data_version, f'{data_type}_ai2.h5'), 'a')
    #                 source_f    = h5py.File(os.path.join('/gallery_millet/chris.kim/data/howto100m', args.data_version, f'{data_type}.h5'), 'r')
    #                 print(f'target filename: {target_f.filename}')
    #                 print(f'source filename: {source_f.filename}')
    #                 for key in tqdm(source_f.keys()):
    #                     target_f.create_dataset(key, data = source_f[key][...])
    #                 target_f.flush()
    #                 target_f.close()

    #         elif 'ai2' not in args.dir_name:
    #             dir_list = ['moma', 'tate', 'getty', 'orsay']
    #             for data_type in DATA_TYPE:
    #                 merged_f = h5py.File(os.path.join('/gallery_millet/chris.kim/data/howto100m', args.data_version, f'{data_type}.h5'), 'a')
    #                 os.chmod(merged_f.filename, mode=0o770)
    #                 print(f'merged_f: {merged_f.filename}')
    #                 for dir_name in dir_list:
    #                     root_dir = os.path.join(LOAD_DIR[dir_name], args.data_version)
    #                     print(f'root_dir: {root_dir}')
    #                     file_list = glob(os.path.join(root_dir, f'{data_type}_part*_*_*.hdf5'))
    #                     file_list.sort()
    #                     pprint(file_list)
    #                     for file_path in tqdm(file_list):
    #                         if (args.exclude_0 == True) and ('_0' in file_path):
    #                             continue
    #                         cur_f = h5py.File(file_path, 'r')
    #                         for key in tqdm(cur_f.keys()):
    #                             merged_f.create_dataset(key, data = cur_f[key][...])
    #                 merged_f.flush()
    #                 merged_f.close()

    #         else:
    #             meta_part_list = [4,5,6,7,8,9]
    #             for data_type in DATA_TYPE:
    #                 merged_f = h5py.File(os.path.join(LOAD_DIR['ai2'], args.data_version, f'{data_type}_ai2.h5'), 'a')
    #                 os.chmod(merged_f.filename, mode=0o770)
    #                 print(f'merged_f: {merged_f.filename}')
    #                 for meta_part in meta_part_list:
    #                     root_dir = os.path.join(LOAD_DIR['ai2'], args.data_version)
    #                     print(f'root_dir: {root_dir}')
    #                     file_list = glob(os.path.join(root_dir, f'{data_type}_*_*_*.hdf5'))
    #                     file_list.sort()
    #                     pprint(file_list)
    #                     for file_path in tqdm(file_list):
    #                         if (args.exclude_0 == True) and ('_0' in file_path):
    #                             continue
    #                         cur_f = h5py.File(file_path, 'r')
    #                         for key in tqdm(cur_f.keys()):
    #                             merged_f.create_dataset(key, data = cur_f[key][...])
    #                 merged_f.flush()
    #                 merged_f.close()
        
    #     elif args.data_version == '370k':
    #         for data_type in DATA_TYPE:
    #             root_dir         = os.path.join(LOAD_DIR['millet'])
    #             target_f         = h5py.File(os.path.join(root_dir, '370k', f'{data_type}.h5'), 'a')
    #             source_file_list = glob(os.path.join(root_dir, '*', f'{data_type}_part*_*_*.hdf5'))
    #             source_file_list.sort()
    #             for source_file in source_file_list:
    #                 if 'part1_8_6' not in source_file:
    #                     print('pass except 186')
    #                     continue
    #                 source_f = h5py.File(os.path.join(source_file), 'r')
    #                 print(f'merge from {source_f.filename} to {target_f.filename}')
    #                 for key in tqdm(source_f.keys()):
    #                     target_f.create_dataset(key, data = source_f[key][...])

    #             target_f.flush()
    #             target_f.close()

    #         pass


    # elif args.dir_name == 'total': # after merge server-wisely
    #     for data_type in DATA_TYPE:
    #         merged_f = h5py.File(os.path.join(LOAD_DIR['moma'], args.data_version, f'{data_type}_{args.dir_name}.hdf5'), 'a')
    #         for dir_name in LOAD_DIR.keys():
    #             root_dir = os.path.join(LOAD_DIR[dir_name], args.data_version)
    #             file_list = glob(os.path.join(root_dir, f'{data_type}_{dir_name}.hdf5'))
    #             print(f'[{data_type}] merge {dir_name} to total')
    #             for file_path in tqdm(file_list):
    #                 cur_f = h5py.File(file_path, 'r')
    #                 for key in tqdm(cur_f.keys()):
    #                     merged_f.create_dataset(key, data = cur_f[key][...])
    #         merged_f.flush()
    #         merged_f.close()

    # elif args.dir_name == 'total_from_scratch':
    #     for data_type in DATA_TYPE:
    #         merged_f = h5py.File(os.path.join(LOAD_DIR['moma'], args.data_version, f'{data_type}_{args.dir_name}.hdf5'), 'a')
    #         for dir_name in LOAD_DIR.keys():
    #             root_dir = os.path.join(LOAD_DIR[dir_name], args.data_version)
    #             file_list = glob(os.path.join(root_dir, f'{data_type}_{dir_name}_*_*.hdf5'))
    #             print(f'[{data_type}] merge {dir_name} to total')
    #             for file_path in tqdm(file_list):
    #                 cur_f = h5py.File(file_path, 'r')
    #                 for key in tqdm(cur_f.keys()):
    #                     merged_f.create_dataset(key, data = cur_f[key][...])
    #         merged_f.flush()
    #         merged_f.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)
    print(colorful.bold_pink("Thank you and Good Job Computer.").styled_string)
