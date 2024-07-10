import os, sys
import argparse
import colorful
from pprint import pprint
from tqdm import tqdm

from glob import glob
import h5py

LOAD_DIR = {
    'ai2'   : '/net/nfs3.prior/dongjook/videocc3m',
    'millet': '/gallery_millet/chris.kim/data/videocc3m/8frames_per_clip',
    'tate'  : '/gallery_tate/dongyeon.woo/jongchan/videocc3m/8frames_per_clip',
    'orsay' : '/gallery_orsay/sangwoo.moon/data/video/cc3m/8frames_per_clip',
    'getty' : '/gallery_getty/dongjoo.kim/vision/cc3m/8frames_per_clip'
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_name",  type=str, required=True,   help="[millet,tate,orsay,getty,ai2]")
    parser.add_argument("--model",     type=str, required=True,   help="[LB, UMT, CLIP_vit_b16, CLIP_vit_l14]")
    parser.add_argument("--merge_type",type=str, required=True,
                        help="[merge_videocc3m_all, merge_single_part]")

    parser.add_argument("--meta_part", type=str, help="after multi-downloading. which part?(0, ..., )")
    parser.add_argument("--part",      type=str, help="for mulit_running. which part?(1, ..., total)")
    parser.add_argument("--total",     type=str, help="for multi_running. how many parts?")

    parser.add_argument("--debug",     action='store_true', help="--debug if you debug")
    return parser.parse_args()


def main(args):
    if not args.merge_type == 'merge_frames':
        if args.model in ['LB', 'CLIP_vit_b16', 'CLIP_vit_l14', 'LB_reverse']:
            DATA_TYPE = [
                # For LB
                #'text_ids',
                'clip_sim',
                'text_emb',
                'clip_emb',
                ]
        elif args.model == 'UMT':
            DATA_TYPE = [
                # For UMT
                'text_ids'
                'clip_sim',
                ]
        else:
            NotImplementedError

    if args.merge_type == 'merge_videocc3m_all':
        """ merge all files based on model
        """
        root_dir = os.path.join(LOAD_DIR[args.dir_name], args.model)
        os.makedirs(root_dir, exist_ok=True)

        for data_type in DATA_TYPE:
            print(data_type)

            source_file_list = []
            source_file_list.extend(glob(os.path.join(LOAD_DIR[args.dir_name], args.model, f'{data_type}_part*_*_*.h5')))
            source_file_list.sort()

            print(f'source_file_list = {len(source_file_list)}')
            if args.debug:
                count = 0
                for source_file in tqdm(source_file_list):
                    source_h5 = h5py.File(source_file, 'r')
                    print(len(source_h5), source_h5.filename)
                    count += len(source_h5)
                print(count)
            else:
                target_file = os.path.join(root_dir, f'{data_type}_total.h5')
                target_h5   = h5py.File(target_file, 'a')
                os.chmod(target_h5.filename, mode=0o777)
                for source_file in tqdm(source_file_list):
                    print(source_file)
                    source_h5 = h5py.File(source_file, 'r')
                    for vid in tqdm(source_h5.keys()):
                        try:
                            target_h5.create_dataset(vid, data = source_h5[vid][...])
                        except Exception as e:
                            print(vid, e)
                print(len(target_h5))

                target_h5.flush()
                target_h5.close()


    if args.merge_type == 'merge_single_part':
        """ merge all files based on meta_total_part
        """
        root_dir = os.path.join(LOAD_DIR[args.dir_name], args.model)
        os.makedirs(root_dir, exist_ok=True)

        for data_type in DATA_TYPE:
            print(data_type)

            source_file_list = []
            source_file_list.extend(glob(os.path.join(root_dir, f'{data_type}_part{args.meta_part}_{args.total}_{args.part}.h5')))
            source_file_list.sort()

            print(f'source_file_list = {len(source_file_list)}')
            if args.debug:
                count = 0
                for source_file in tqdm(source_file_list):
                    source_h5 = h5py.File(source_file, 'r')
                    print(len(source_h5), source_h5.filename)
                    count += len(source_h5)
                print(count)
            else:
                target_file = os.path.join(root_dir, f'{data_type}_total.h5')
                target_h5   = h5py.File(target_file, 'a')
                os.chmod(target_h5.filename, mode=0o777)
                for source_file in tqdm(source_file_list):
                    print(source_file)
                    source_h5 = h5py.File(source_file, 'r')
                    for vid in tqdm(source_h5.keys()):
                        try:
                            target_h5.create_dataset(vid, data = source_h5[vid][...])
                        except Exception as e:
                            print(vid, e)
                print(len(target_h5))

                target_h5.flush()
                target_h5.close()

    elif args.merge_type == 'merge_all_to_total':
        """ merge all files based on model
        """
        assert args.data_version == 'total'
        root_dir = os.path.join(LOAD_DIR['millet'], 'total', args.model)
        for data_type in DATA_TYPE:
            source_file_list = []
            source_file_list.extend(glob(os.path.join(LOAD_DIR['moma'], 'subset',    args.model, f'{data_type}_*.h5')))
            source_file_list.extend(glob(os.path.join(LOAD_DIR['moma'], 'valid',     args.model, f'{data_type}_*.h5')))
            source_file_list.extend(glob(os.path.join(LOAD_DIR['moma'], 'leftovers', args.model, f'{data_type}_*.h5')))
            if args.debug:
                target_file = os.path.join(root_dir, f'{data_type}_total.h5')
                target_h5  = h5py.File(target_file, 'r')
                print(len(target_h5), target_h5.filename)
                pprint(source_file_list)
            else:
                target_file = os.path.join(root_dir, f'{data_type}_total.h5')
                target_h5  = h5py.File(target_file, 'a')
                os.chmod(target_h5.filename, mode=0o777)
                for source_file in source_file_list:
                    source_h5 = h5py.File(source_file, 'r')
                    for vid in tqdm(source_h5.keys()):
                        try:
                            target_h5.create_dataset(vid, data = source_h5[vid][...])
                        except Exception as e:
                            pass # subset-valid duplication
                print(f'total length: {len(target_h5)}')
                target_h5.flush()
                target_h5.close()

    elif args.merge_type == 'merge_valid_to_subset':
        """ merge all files based on model
        """
        assert args.data_version == 'subset'
        root_dir = os.path.join(LOAD_DIR['millet'], 'subset', args.model)
        for data_type in DATA_TYPE:
            source_file_list = []
            source_file_list.extend(glob(os.path.join(LOAD_DIR['moma'], 'subset',    args.model, f'{data_type}_*.h5')))
            source_file_list.extend(glob(os.path.join(LOAD_DIR['moma'], 'valid',     args.model, f'{data_type}_*.h5')))
            if args.debug:
                target_file = os.path.join(root_dir, f'{data_type}_total.h5')
                #target_h5  = h5py.File(target_file, 'r')
                #print(len(target_h5), target_h5.filename)
                pprint(source_file_list)
            else:
                target_file = os.path.join(root_dir, f'{data_type}_total.h5')
                target_h5  = h5py.File(target_file, 'a')
                os.chmod(target_h5.filename, mode=0o777)
                for source_file in source_file_list:
                    source_h5 = h5py.File(source_file, 'r')
                    for vid in tqdm(source_h5.keys()):
                        try:
                            target_h5.create_dataset(vid, data = source_h5[vid][...])
                        except Exception as e:
                            pass # subset-valid duplication
                print(f'total length: {len(target_h5)}')
                target_h5.flush()
                target_h5.close()

    elif args.merge_type == 'merge_frames':
        """ merge all files based on model
        """
        root_dir = os.path.join(LOAD_DIR['millet'])
        source_file_list = []
        source_file_list.extend(glob(os.path.join(LOAD_DIR['millet'], 'preprocessed_frames_part*.h5')))
        source_file_list.extend(glob(os.path.join(LOAD_DIR['tate'],   'preprocessed_frames_part*.h5')))
        source_file_list.extend(glob(os.path.join(LOAD_DIR['orsay'],  'preprocessed_frames_part*.h5')))
        source_file_list.extend(glob(os.path.join(LOAD_DIR['getty'],  'preprocessed_frames_part*.h5')))

        if args.debug:
            target_file = os.path.join('/gallery_millet/chris.kim/data/videocc3m/8frames_per_clip_merged', f'preprocessed_frames_total.h5')
            #target_h5  = h5py.File(target_file, 'r')
            #print(len(target_h5), target_h5.filename)
            pprint(source_file_list)
        else:
            target_file = os.path.join('/gallery_millet/chris.kim/data/videocc3m/8frames_per_clip_merged', f'preprocessed_frames_total.h5')
            target_h5  = h5py.File(target_file, 'a')
            os.chmod(target_h5.filename, mode=0o777)
            for source_file in tqdm(source_file_list):
                print(f'{source_file} to {target_file}')
                source_h5 = h5py.File(source_file, 'r')
                for vid in tqdm(source_h5.keys()):
                    try:
                        target_h5.create_dataset(vid, data = source_h5[vid][...])
                    except Exception as e:
                        pass # subset-valid duplication
            print(f'total length: {len(target_h5)}')
            target_h5.flush()
            target_h5.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)
    print(colorful.bold_pink("Thank you and Good Job Computer.").styled_string)
