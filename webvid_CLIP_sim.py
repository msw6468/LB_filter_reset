# Ignore warning
import warnings
warnings.filterwarnings(action='ignore')

import os
import json
import pickle as pkl
import h5py
from pathlib import Path
from abc import ABC
import io

import numpy as np
import pandas as pd

import torch
import torch.multiprocessing
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from PIL import Image

import argparse
import colorful
from tqdm import tqdm
from pprint import pprint

# For LanguageBind
from languagebind import LanguageBind, LanguageBindImageTokenizer

# For CLIP
from clip.simple_tokenizer import SimpleTokenizer
import clip
from einops import rearrange, repeat

# from languagebind import to_device, transform_dict

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275,  0.40821073)
OPENAI_DATASET_STD  = (0.26862954, 0.26130258, 0.27577711)

LOAD_DIR = {
    'millet': '/gallery_millet/chris.kim/data/webvid',
    'tate':   '/gallery_tate/dongyeon.woo/jongchan/webvid',
    'getty':  '/gallery_getty/dongjoo.kim/vision/webvid',
    'moma':   '/gallery_moma/jongchan.noh/webvid',
    'orsay':  '/gallery_orsay/sangwoo.moon/JC/webvid',
}

# utils ----------------------------------------------------------
# def get_LB_model(args):
#     clip_type = {'video': 'LanguageBind_Video_FT',}  # also LanguageBind_Video
#     cache_dir = '/net/nfs3.prior/dongjook/Language_Bind_cache' if args.dir_name == 'ai2' else './cache_dir'
#     model     = LanguageBind(clip_type=clip_type, cache_dir=cache_dir)
#     model     = model.to(args.device)
#     pretrained_ckpt = 'LanguageBind/LanguageBind_Image'
#     tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt,
#                                                         cache_dir=f'{cache_dir}/tokenizer_cache_dir')
#     transform = transforms.Compose(
#         [
#             transforms.ToTensor(),
#             transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
#             transforms.CenterCrop(224),
#             transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)  # assume image
#         ]
#     )
#     model.eval()
#     return model, transform, tokenizer

def get_CLIP_model(args):
    # model_path = '/net/nfs3.prior/dongjook/Language_Bind_cache' if args.dir_name == 'ai2' else './cache_dir'
    if args.clip_type =='vit_l14':
        model_path = '/gallery_moma/sangwoo.moon/.cache/clip/ViT-L-14.pt'
    elif args.clip_type =='vit_b16':
        model_path = '/gallery_moma/sangwoo.moon/.cache/clip/ViT-B-16.pt'
    else:
        NotImplementedError

    model, _   = clip.load(model_path, device=args.device)
    tokenizer = SimpleTokenizer()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)  # assume image

        ]
    )
    model.eval()
    return model, transform, tokenizer

def get_total_video_dict(args):
    file_name = f'{LOAD_DIR["millet"]}/webvid10m_train_{args.dir_name}_w_uid_part{args.meta_part}.csv'
    total_video_dict = pd.read_csv(file_name, index_col=0)
    return total_video_dict

def get_preprocessed_frames_hdf5(args):
    h5_filename = os.path.join(LOAD_DIR[args.dir_name], f'preprocessed_frames_{args.dir_name}_{args.meta_part}.h5')
    h5_file = h5py.File(h5_filename, 'r')
    return h5_file


def get_partitioned_dict(total_video_dict, total, part):
    if total==1:
        return total_video_dict

    if args.total > 1:
        total_video_list = list(total_video_dict)
        total_size       = len(total_video_dict)
        part_size        = int(total_size / total)

        start            = part_size * (part - 1)
        end              = part_size * (part) if part < total else total_size

        new_total_video_dict  = total_video_dict.iloc[start:end]
        print(f'[PARTITION] Total datasets  : {len(total_video_dict)}, Part: {part}/{total} [{start}:{end}]')
        print(f'[PARTITION] After partition : {len(new_total_video_dict)}')

    return new_total_video_dict


def get_preprocessed_frames_hdf5(args):
    h5_filename = os.path.join(LOAD_DIR[args.dir_name], f'preprocessed_frames_{args.dir_name}_{args.meta_part}.h5')
    h5_file = h5py.File(h5_filename, 'r')
    return h5_file


def get_h5py_files(args):
    h5py_f = {}
    # h5py_f['text_ids_h5'] = h5py.File(os.path.join(args.save_path, f'text_ids_part{args.meta_part}_{args.total}_{args.part}.h5'), 'a')
    h5py_f['text_emb_h5'] = h5py.File(os.path.join(args.save_path, f'text_emb_part{args.dir_name}{args.meta_part}_{args.total}_{args.part}.h5'), 'a')
    h5py_f['clip_emb_h5'] = h5py.File(os.path.join(args.save_path, f'clip_emb_part{args.dir_name}{args.meta_part}_{args.total}_{args.part}.h5'), 'a')
    h5py_f['clip_sim_h5'] = h5py.File(os.path.join(args.save_path, f'clip_sim_part{args.dir_name}{args.meta_part}_{args.total}_{args.part}.h5'), 'a')

    for key in h5py_f:
        h5py_f[key].flush()
        os.chmod(h5py_f[key].filename, mode=0o777)

    return h5py_f

# ----------------------------------------------------------------

# ================
# Generic Datasets
# ================
class BaseDataset(Dataset, ABC):
    name = 'base'
    dataset_size = 0

    def __init__(self):
        super().__init__()

    def __len__(self):
        return self.dataset_size

    def collate_fn(self, batch):
        return default_collate(batch)

# ================
# WebVid  Datasets
# ================
class WebVid(BaseDataset):
    name = 'howto100m'
    def __init__(self, args, tokenizer, processor, frames_h5=None, video_dict=None):
        super(WebVid, self).__init__()

        self.args   = args
        self.debug  = args.debug
        self.device = args.device

        self.max_frames = args.max_frames
        self.frame_idxs = [0,1,2,3,4,5,6,7] if self.max_frames==8 else [0,2,4,6] # UMT case
        self.max_words  = 77

        self.frames_h5 = frames_h5

        # Set preprocess
        self.tokenizer = tokenizer
        self.processor = processor

        # Set dataframe
        self.df = video_dict


    def _get_frames(self, video_id=None, text_id=None):
        """ Get video information
        INPUT:
            video_id: video_id
            text_id : text_id
        OUTPUT:
            clip_images: (3, max_frame, 224, 224), torch.tensor
                - max_frame : max frame per clip
        """
        #====================
        # Get all frames
        #====================
        if self.args.frame_load == 'hdf5':
            try:
                # Load from frame -----------------------------------------
                binary_images = self.frames_h5[video_id][...]
                images = []
                for binary_image in binary_images:
                    images.append(self.processor(Image.open(io.BytesIO(binary_image))))
                images = torch.stack(images)
                images = images.permute(1, 0, 2, 3) # (T, H, W, C) -> (C, T, H, W)
                # images = images.permute(3, 0, 1, 2) # (T, H, W, C) -> (C, T, H, W)

                # previous code
                # images = torch.zeros((self.max_frames, 224, 224, 3))
                # for i, binary_image in enumerate(binary_images):
                #     images[i] = torch.from_numpy(np.array(Image.open(io.BytesIO(binary_image))))
                # # Checking code!
                # # Image.open(io.BytesIO(images[0])).save(f'temp/temp.jpg')
                # images = images.permute(3, 0, 1, 2) # (T, H, W, C) -> (C, T, H, W)
                # images = self.processor(images)
                # ----------------------------------------------------------

            except Exception as e:
                print(f'video_id {video_id}, text_id {text_id} sample is corrupted, {e}')
                return torch.zeros((3, self.max_frames, 224, 224), dtype=torch.float), False # bad clip-captions
        else:
            NotImplementedError

        return images, True


    def __repr__(self):
        return str(self)


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        data               = self.df.iloc[idx]
        unique_id          = data['unique_id'] # use idx as unique identifier (name for dataset in h5 file)
        video_id           = f"{data['page_dir']}/{data['videoid']}" 
        raw_text           = data['name']
        frames, valid_flag = self._get_frames(video_id=video_id)

        return unique_id, frames, raw_text, valid_flag


def save_embeds_sims_chunk(args,
                        # text_ids_dict,
                        text_emb_dict,
                        clip_emb_dict,
                        h5py_f):

    # Save as single video id
    for video_id in clip_emb_dict.keys():
        for key in h5py_f: # To overwrite
            if video_id in h5py_f[key].keys():
                del h5py_f[key][video_id]
        # h5py_f['text_ids_h5'].create_dataset(video_id, data = text_ids_dict[video_id])
        h5py_f['text_emb_h5'].create_dataset(video_id, data = text_emb_dict[video_id])
        h5py_f['clip_emb_h5'].create_dataset(video_id, data = clip_emb_dict[video_id])
        similarity = clip_emb_dict[video_id] @ text_emb_dict[video_id].T
        h5py_f['clip_sim_h5'].create_dataset(video_id, data = similarity)

    # Flush
    for key in h5py_f:
        h5py_f[key].flush()

    # Save flag after flush
    if not args.debug:
        for video_id in clip_emb_dict.keys():
            flag_save_path = os.path.join(args.flag_dir, f'{video_id}')
            Path(flag_save_path).touch()
    else:
        # from IPython import embed; embed(colors='neutral')  # XXX DEBUG  # yapf: disable
        pass

    return


# %%
def parse_args():
    parser = argparse.ArgumentParser()
    # For partition
    parser.add_argument("--dir_name",      type=str, default='moma', help="[moma, tate, getty, orsay, ai2]")
    parser.add_argument("--meta_part",     type=int, required=True, help="after multi-downloading. which part?(0, ..., )")

    parser.add_argument("--part",      type=int, default=1,      help="for mulit_running. which part?(1, ..., total)")
    parser.add_argument("--total",     type=int, default=1,      help="for multi_running. how many parts?")

    # For dataloader
    parser.add_argument("--device",      type=str, default='cuda')
    parser.add_argument("--batch_size",  type=int, default=200, help="[100 for 24GB, 200 for 48GB]")
    parser.add_argument("--num_workers", type=int, default=4)

    # others
    parser.add_argument("--num_segment",  type=int, default=50)

    parser.add_argument("--debug",      type=str, default='False')
    parser.add_argument("--frame_load", type=str, default='hdf5', help='[image, hdf5]')
    parser.add_argument("--max_frames", type=int, default=8, help='[4,  8]')
    parser.add_argument("--final_check", type=str, default='False', help='[True, False]')
    parser.add_argument("--clip_type", type=str, default='vit_l14', help='[vit_l14, vit_b16]')

    return parser.parse_args()

# %%
def main(args):
    args.debug = True if args.debug == 'True' else False
    args.final_check = True if args.final_check == 'True' else False
    args.root_path = os.path.join(LOAD_DIR['orsay'])
    args.save_path = os.path.join(args.root_path, f'CLIP_{args.clip_type}') if not args.debug else os.path.join(args.root_path, f'CLIP_{args.clip_type}', 'debug')

    pprint(args)
    print(f'Debug mode: {args.debug}')

    print('Load model')
    # model, transform, tokenizer = get_LB_model(args)
    model, transform, tokenizer = get_CLIP_model(args)
    SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    print('Load json(total_video_dict)')
    total_video_dict = get_total_video_dict(args)
    print(f'Load json(total_video_dict) results: Number of videos = {len(total_video_dict)}')

    print(f'[PARTITION] Select data {args.part}/{args.total}')
    total_video_dict = get_partitioned_dict(total_video_dict, args.total, args.part)

    print(f'Set save path on {args.save_path}')
    args.flag_dir   = os.path.join(args.save_path, 'final_flag')
    os.makedirs(args.flag_dir, exist_ok=True, mode=0o777)
    h5py_f = get_h5py_files(args)

    print(f'Load preprocessed_frames')
    frames_h5 = get_preprocessed_frames_hdf5(args)

    # remove processed indexs
    if args.final_check:
        print(f'Final check processed frames removal')
        processed_index_list = list(h5py_f['clip_sim_h5'].keys())
        processed_index_list = list(map(int, processed_index_list))
    else:
        processed_index_list = os.listdir(args.flag_dir)
        processed_index_list = list(map(int, processed_index_list))
    total_video_dict   = total_video_dict[~total_video_dict['unique_id'].isin(processed_index_list)]
    print(f'After remove processed indexes : {len(total_video_dict)}')

    dataset  = WebVid(
        args       = args,
        tokenizer  = tokenizer,
        processor  = transform,
        video_dict = total_video_dict,
        frames_h5  = frames_h5,)

    dataloader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        # collate_fn  = dataset.collate_fn,
        drop_last   = False,
        pin_memory  = True,  # better when training on GPU.
        shuffle     = False) # Don't need to shuffle for captioning


    print('Start batch')
    step = 0
    with torch.no_grad():
        model.eval()
        pbar = tqdm(dataloader)
        for batch in pbar:
            pbar.set_description(f"[{args.dir_name}_{args.meta_part}[{args.part:2d}/{args.total:2d}]")
            unique_ids, frames, raw_texts, valid_flag = batch
            if np.sum(np.array(valid_flag)) == 0:
                continue # ignore current batch when all clips are bad
            unique_ids = np.array(unique_ids)[np.array(valid_flag)]
            raw_texts = list(np.array(raw_texts)[np.array(valid_flag)])
            # texts     = dataset.tokenizer(raw_texts,
            #                         max_length=dataset.max_words,
            #                         padding='max_length',
            #                         truncation=True,
            #                         return_tensors='pt')
            # texts['input_ids']      = texts['input_ids'].to(args.device)
            # texts['attention_mask'] = texts['attention_mask'].to(args.device)
            # ---------------------------------------------
            max_words = 77
            n_caption  = len(raw_texts)
            text_info = np.zeros((n_caption, 77), dtype=np.longlong)
            text_mask = np.zeros((n_caption, 77), dtype=int)

            for i, text in enumerate(raw_texts):
                words = dataset.tokenizer.tokenize(text)
                words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
                total_length_with_CLS = max_words - 1
                if len(words) > total_length_with_CLS:
                    words = words[:total_length_with_CLS]
                words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]

                input_ids   = tokenizer.convert_tokens_to_ids(words)
                input_mask  = [1] * len(input_ids)
                while len(input_ids) < max_words:
                    input_ids.append(0)
                    input_mask.append(0)

                assert len(input_ids)   == max_words
                assert len(input_mask)  == max_words

                text_info[i] = np.array(input_ids)
                text_mask[i] = np.array(input_mask)
            # ---------------------------------------------
            frames    = frames[valid_flag].to(args.device)
            text_info = torch.tensor(text_info).to(args.device)
            # ---------------------------------------------
            frames    = rearrange(frames, 'B C F H W -> (B F) C H W')
            frame_emb = model.encode_image(frames)
            frame_emb = frame_emb / frame_emb.norm(dim=1, keepdim=True)
            text_emb  = model.encode_text(text_info)
            text_emb  = text_emb / text_emb.norm(dim=1, keepdim=True)

            clip_emb   = rearrange(frame_emb, '(B F) feature_dim -> B F feature_dim', F = 8) # (B, F, 512)
            text_emb   = rearrange(text_emb, 'B feature_dim -> B feature_dim 1') # -> (B, F, 1)
            similarity = torch.bmm(clip_emb, text_emb).squeeze(2) # -> (B, F)

            clip_emb   = clip_emb.detach().cpu().numpy()
            text_emb   = text_emb.detach().cpu().numpy()
            similarity = similarity.detach().cpu().numpy()

            for u_id, c_emb, t_emb, sim in zip(unique_ids, clip_emb, text_emb, similarity):
                if (str(u_id) in h5py_f['clip_sim_h5'].keys()):
                    del h5py_f['clip_sim_h5'][str(u_id)]
                if (str(u_id) in h5py_f['text_emb_h5'].keys()):
                    del h5py_f['text_emb_h5'][str(u_id)]
                if (str(u_id) in h5py_f['clip_emb_h5'].keys()):
                    del h5py_f['clip_emb_h5'][str(u_id)]

                h5py_f['text_emb_h5'].create_dataset(str(u_id), data = t_emb)
                h5py_f['clip_emb_h5'].create_dataset(str(u_id), data = c_emb)
                h5py_f['clip_sim_h5'].create_dataset(str(u_id), data = np.array([[sim]]))
                h5py_f['text_emb_h5'].flush()
                h5py_f['clip_emb_h5'].flush()
                h5py_f['clip_sim_h5'].flush()
                flag_save_path = os.path.join(args.flag_dir, f'{u_id}')
                Path(flag_save_path).touch()

            step += 1

    
    for key in h5py_f.keys():
        h5py_f[key].close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print(colorful.bold_pink("Thank you and Good Job Computer.").styled_string)
