from functools import partial
from tqdm import tqdm
import torch
import json
import os
import pickle
from sae_lens import SAE, ActivationsStore
import numpy as np
from pathlib import Path

def find_max_activation(model, sae, activation_store, feature_idx, num_batches=1):
    max_activation = 0.0
    for _ in tqdm(range(num_batches), desc="Finding max activation"):
        tokens = activation_store.get_batch_tokens()
        _, cache = model.run_with_cache(
            tokens,
            stop_at_layer=sae.cfg.hook_layer + 1,
            names_filter=[sae.cfg.hook_name],
        )
        sae_in = cache[sae.cfg.hook_name]
        feature_acts = sae.encode(sae_in).squeeze()
        feature_acts = feature_acts.flatten(0, 1)
        batch_max = feature_acts[:, feature_idx].max().item()
        max_activation = max(max_activation, batch_max)
    return max_activation

def steering_hook_fn(activations, hook, steering_strength, steering_vector, max_act):
    return activations + max_act * steering_strength * steering_vector

def ablate_feature_hook_fn(feature_activations, hook, feature_ids, position=None):
    if position is None:
        feature_activations[:, :, feature_ids] = 0
    else:
        feature_activations[:, position, feature_ids] = 0
    return feature_activations

def generate_with_steering(model, sae, prompt, feature_idx, max_act, strength=1.0, crop=False):
    input_ids = model.to_tokens(prompt, prepend_bos=sae.cfg.prepend_bos)
    steering_vector = sae.W_dec[feature_idx].to(model.cfg.device)

    if strength != 0.0:
        hook = partial(steering_hook_fn, steering_strength=strength, steering_vector=steering_vector, max_act=max_act)
    else:
        hook = partial(ablate_feature_hook_fn, feature_ids=feature_idx)

    with model.hooks(fwd_hooks=[(sae.cfg.hook_name, hook)]):
        output = model.generate(
            input_ids,
            max_new_tokens=95,
            temperature=0.7,
            top_p=0.9,
            stop_at_eos=False,
            prepend_bos=sae.cfg.prepend_bos,
        )
    if crop:
        return model.tokenizer.decode(output[0][input_ids.shape[1]:])
    return model.tokenizer.decode(output[0])

def run_steering_experiment(model, prompts, top_features_per_layer, layers, output_folder, device):
    results = {}
    for layer in layers:
        print(f"\n=== Layer {layer} ===")
        results[layer] = {}
        sae, _, _ = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res-canonical",
            sae_id=f"layer_{layer}/width_16k/canonical"
        )
        sae = sae.to(device)
        activation_store = ActivationsStore.from_sae(
            model=model,
            sae=sae,
            streaming=True,
            store_batch_size_prompts=8,
            train_batch_size_tokens=4096,
            n_batches_in_buffer=32,
            device=device,
        )

        top_features = top_features_per_layer[layer]
        for j, (feature_id, _) in enumerate(top_features):
            results[layer][feature_id] = {}
            for l, prompt in enumerate(prompts):
                results[layer][feature_id][prompt] = {}

                max_act = find_max_activation(model, sae, activation_store, feature_id)
                normal_text = model.generate(prompt, max_new_tokens=95, stop_at_eos=False, prepend_bos=sae.cfg.prepend_bos)
                results[layer][feature_id][prompt]['original'] = normal_text

                for strength in [-4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
                    steered_text = generate_with_steering(model, sae, prompt, feature_id, max_act, strength, crop=(l > 29))
                    results[layer][feature_id][prompt][strength] = steered_text

                out_path = os.path.join(output_folder, f"generated_texts_layer_{layer}_feature_{feature_id}_{j}_prompt_{l}.json")
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=4)
    return results


def get_top_features_from_xgboost(model_name, place, width, layers, feature_type, top_n=10):
    top_features_per_layer = {}
    for layer in layers:
        path = Path(__file__).resolve().parent.parent / f"models/{model_name}-{place}-{width}_layer{layer}_{feature_type}_xgboost.pkl"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as f:
            clf = pickle.load(f)

        booster = clf.get_booster()
        gain_dict = booster.get_score(importance_type="gain")

        # Extract feature importances like "f0", "f123", etc.
        sorted_feats = sorted(
            ((int(feat[1:]), score) for feat, score in gain_dict.items()),
            key=lambda x: x[1],
            reverse=True
        )
        top_features_per_layer[layer] = sorted_feats[:top_n]
    return top_features_per_layer