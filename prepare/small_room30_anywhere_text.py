"""Encode the actual added sentence using the same hash-pinned CLIP weights."""
import inspect
from pathlib import Path
import numpy as np
from small_room30_adm_dataset import sha
from small_room30_anywhere_data import ANYWHERE,TEXTS
from train_small_room30_student_v2_compat import load_baseline_without_pickle

CLIP_SHA='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'


def encode_texts(path,texts):
    if not path.is_file():raise FileNotFoundError('Existing ViT-B-32.pt required: --clip-weights PATH; no download performed')
    if sha(path)!=CLIP_SHA:raise ValueError('CLIP checkpoint differs from audited ViT-B/32')
    import torch
    try:
        import clip
        from clip.model import build_model
    except ImportError as exc:
        raise ImportError('Use the original afford environment with OpenAI CLIP installed; no automatic install is performed') from exc
    # Official hash-pinned JIT archive only. Never fall back to torch.load.
    scripted=torch.jit.load(str(path),map_location='cpu').eval()
    model=build_model(scripted.state_dict()).float().eval();del scripted
    for param in model.parameters():param.requires_grad_(False)
    with torch.no_grad():
        vectors=model.encode_text(clip.tokenize(texts).long()).float()
        vectors=torch.nn.functional.normalize(vectors,dim=-1).cpu()
    if vectors.shape!=(len(texts),512) or not torch.isfinite(vectors).all():raise ValueError('Invalid CLIP features')
    provenance=dict(clip_checkpoint_sha256=CLIP_SHA,device='cpu',dtype='float32',
        model_source_sha256=sha(Path(inspect.getfile(build_model))),
        tokenizer_source_sha256=sha(Path(inspect.getfile(clip.tokenize))),
        method='Hash-pinned official JIT state -> CLIP eager model; no arbitrary checkpoint pickle fallback')
    return vectors,provenance


def prepare_text(baseline,legacy_binding,clip_weights):
    old,text,clip_sha=load_baseline_without_pickle(baseline,legacy_binding)
    keys=sorted(TEXTS)
    values,provenance=encode_texts(clip_weights,[TEXTS[k] for k in keys])
    differences={}
    for key,prior in text.items():
        new=values[keys.index(key):keys.index(key)+1]
        cosine=float((prior*new).sum());delta=float((prior-new).abs().max())
        if cosine<.999 or delta>.005:raise ValueError('Encoded legacy text does not match prior frozen CLIP: '+key)
        differences[key]=dict(cosine=cosine,max_abs_delta=delta)
    # Preserve previously used four vectors exactly, add the genuinely encoded fifth.
    text[ANYWHERE]=values[keys.index(ANYWHERE):keys.index(ANYWHERE)+1].clone()
    if np.allclose(text[ANYWHERE].numpy(),0):raise ValueError('Placeholder anywhere vector rejected')
    provenance.update(prompt_texts=TEXTS,legacy_vector_comparison=differences,
        legacy_vectors_kept_exact=True,new_feature_is_real_clip_encoding=True)
    return text,provenance
