import torch
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer

if __name__ == '__main__':
    device = 'cuda:0'
    device = torch.device(device)
    clip_type = {
        'video': 'LanguageBind_Video_FT',  # also LanguageBind_Video
    }

    model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
    model = model.to(device)
    model.eval()
    pretrained_ckpt = f'LanguageBind/LanguageBind_Image'
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}

    video = ['assets/video/0.mp4', 'assets/video/1.mp4']
    language = ["Training a parakeet to climb up a ladder.", 'A lion climbing a tree to catch a monkey.']

    # for i in range(7):
    #     video.extend(video)
    #     language.extend(language)
    # video = video[:200]
    # language = language[:200]
    # print(len(video), len(language))

    inputs = {
        'video': to_device(modality_transform['video'](video), device),
    }
    inputs['language'] = to_device(tokenizer(language, max_length=77, padding='max_length',
                                             truncation=True, return_tensors='pt'), device)

    with torch.no_grad():
        embeddings = model(inputs)
    print(embeddings['video'][0].sum())
    print(embeddings['language'][0].sum())

    print("Video x Text: \n",
          torch.softmax(embeddings['video'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())


    with torch.no_grad():
        embeddings = model(inputs)
    print(embeddings['video'][0].sum())
    print(embeddings['language'][0].sum())

    print("Video x Text: \n",
          torch.softmax(embeddings['video'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())

