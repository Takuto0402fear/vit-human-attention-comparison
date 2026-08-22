"""
Builds attribute_mapping.json and prompt_bank.json for the OSIE
text-alignment pilot. No GPU, no CLIP model, no randomness beyond the
templates' own fixed order.

attribute_mapping.json records the CANONICAL attrs.mat attribute order/names
as read directly from data/attrs.mat (via h5py; verified in the earlier
attribute-grounding pilot's scripts/verify_osie_attribute_coords.py and
scripts/run_osie_attribute_grounding_pilot.py), mapped to this task's
human-facing attribute names (Face/Emotion/Touched/Gazed/Motion/Sound/
Smell/Taste/Touch/Text/Watchability/Operability). "Touch" and "Touched" are
kept as two distinct attrs.mat fields ("touch" and "touched") -- never
merged or confused.

prompt_bank.json expands each attribute's fixed base phrase through 16
fixed templates (192 sentences total), plus a separate "raw_label" baseline
(the attribute name alone) NOT mixed into the main ensemble. No LLM/VLM is
used to generate or rewrite any sentence -- every sentence is a plain
Python str.format() of a hardcoded template and a hardcoded phrase.

Writes only: outputs/osie_text_alignment_pilot/attribute_mapping.json,
outputs/osie_text_alignment_pilot/prompt_bank.json.

Usage (PowerShell):
    python scripts\\build_osie_text_alignment_prompts.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import json
import os

import h5py

# ======================= CONFIG =======================
DATA_BASE = r"C:\Users\user\gaze\datasets\osie\predicting-human-gaze-beyond-pixels"
ATTRS_PATH = os.path.join(DATA_BASE, "data", "attrs.mat")

OUT_DIR = r"C:\Users\user\gaze\outputs\osie_text_alignment_pilot"
ATTRIBUTE_MAPPING_JSON = os.path.join(OUT_DIR, "attribute_mapping.json")
PROMPT_BANK_JSON = os.path.join(OUT_DIR, "prompt_bank.json")

# attrs.mat field name (verified via h5py) -> this task's human-facing name.
MAT_NAME_TO_TASK_NAME = {
    "text": "Text",
    "face": "Face",
    "emotion": "Emotion",
    "sound": "Sound",
    "smell": "Smell",
    "taste": "Taste",
    "touch": "Touch",
    "motion": "Motion",
    "operability": "Operability",
    "watchability": "Watchability",
    "touched": "Touched",
    "gazed": "Gazed",
}

# Fixed base phrase per task attribute name (verbatim from the task spec).
BASE_PHRASES = {
    "Face": "a face",
    "Emotion": "a face showing a clear emotion",
    "Touched": "an object being touched by a person or animal",
    "Gazed": "an object being looked at by a person or animal",
    "Motion": "an object or living being showing motion",
    "Sound": "an object producing sound",
    "Smell": "an object with a noticeable smell",
    "Taste": "food or drink that can be tasted",
    "Touch": "an object with a distinctive tactile quality",
    "Text": "written text, letters, or numbers",
    "Watchability": "an object designed to be watched",
    "Operability": "an object designed to be operated by hand",
}

# 16 fixed templates, verbatim from the task spec, index 0..15.
TEMPLATES = [
    "{phrase}",
    "a photo of {phrase}",
    "an image of {phrase}",
    "a photograph of {phrase}",
    "a photo containing {phrase}",
    "an image containing {phrase}",
    "a photograph containing {phrase}",
    "a natural scene containing {phrase}",
    "a real-world scene containing {phrase}",
    "an image region containing {phrase}",
    "a region of the image containing {phrase}",
    "a visual region showing {phrase}",
    "a part of the image showing {phrase}",
    "a cropped image region containing {phrase}",
    "a visible example of {phrase}",
    "{phrase} visible in an image",
]
# ======================================================


def h5_char_str(f, ref):
    arr = f[ref][()]
    return "".join(chr(int(c)) for c in arr.flatten())


def load_attrs_mat_order():
    """Returns the attrs.mat attrNames order exactly as stored -- read-only."""
    with h5py.File(ATTRS_PATH, "r") as f:
        names_ds = f["attrNames"]
        return [h5_char_str(f, names_ds[i, 0]) for i in range(names_ds.shape[0])]


def build_attribute_mapping():
    mat_order = load_attrs_mat_order()
    if set(mat_order) != set(MAT_NAME_TO_TASK_NAME.keys()):
        raise RuntimeError(
            f"STOP: attrs.mat attrNames {mat_order} do not exactly match the "
            f"expected set {sorted(MAT_NAME_TO_TASK_NAME.keys())} -- do not "
            "guess a mapping, investigate the mismatch first.")

    task_names = set(MAT_NAME_TO_TASK_NAME.values())
    if len(task_names) != 12 or "Touch" not in task_names or "Touched" not in task_names:
        raise RuntimeError(
            f"STOP: expected 12 distinct task attribute names including both "
            f"'Touch' and 'Touched', got {sorted(task_names)}")
    if MAT_NAME_TO_TASK_NAME["touch"] == MAT_NAME_TO_TASK_NAME["touched"]:
        raise RuntimeError("STOP: 'touch' and 'touched' must map to distinct task names")

    return {
        "attrs_mat_path": ATTRS_PATH,
        "attrs_mat_order_as_stored": mat_order,
        "attrs_mat_order_zero_based_index": {name: i for i, name in enumerate(mat_order)},
        "mat_name_to_task_name": MAT_NAME_TO_TASK_NAME,
        "task_name_to_mat_index": {
            MAT_NAME_TO_TASK_NAME[name]: i for i, name in enumerate(mat_order)
        },
        "task_attribute_order": [MAT_NAME_TO_TASK_NAME[name] for name in mat_order],
        "note": (
            "'touch' (mat) -> 'Touch' (task) and 'touched' (mat) -> 'Touched' "
            "(task) are kept as two distinct attrs.mat fields at all times; "
            "never merged."
        ),
    }


def build_prompt_bank():
    if set(BASE_PHRASES.keys()) != set(MAT_NAME_TO_TASK_NAME.values()):
        raise RuntimeError("STOP: BASE_PHRASES keys do not match the 12 task attribute names")
    if len(TEMPLATES) != 16:
        raise RuntimeError(f"STOP: expected 16 templates, got {len(TEMPLATES)}")

    bank = {"templates": TEMPLATES, "base_phrases": BASE_PHRASES, "attributes": {}}
    seen_all_sentences = set()
    for attr, phrase in BASE_PHRASES.items():
        sentences = []
        for idx, tmpl in enumerate(TEMPLATES):
            sentence = tmpl.format(phrase=phrase)
            if not sentence or not sentence.strip():
                raise RuntimeError(f"STOP: empty sentence for attribute={attr}, template={idx}")
            # Verbatim template.format(phrase=phrase) -- no capitalization or
            # other rewriting is applied; the task spec's templates/phrases
            # are used exactly as given.
            sentences.append({"prompt_index": idx, "template": tmpl, "sentence": sentence})
            key = (attr, sentence)
            if key in seen_all_sentences:
                raise RuntimeError(f"STOP: duplicate sentence within bank: {key}")
            seen_all_sentences.add(key)
        bank["attributes"][attr] = {
            "base_phrase": phrase,
            "prompts": sentences,
            "raw_label_baseline": attr,  # NOT mixed into the main ensemble
        }

    # Cross-attribute duplicate sentence check (e.g. two attributes producing
    # the identical rendered sentence would silently corrupt the text-vector
    # <-> attribute correspondence).
    flat = [(attr, p["sentence"]) for attr, data in bank["attributes"].items() for p in data["prompts"]]
    sentence_only = [s for _, s in flat]
    if len(sentence_only) != len(set(sentence_only)):
        dupes = {s for s in sentence_only if sentence_only.count(s) > 1}
        raise RuntimeError(f"STOP: identical sentence rendered for >1 attribute: {dupes}")

    expected_total = 12 * 16
    if len(flat) != expected_total:
        raise RuntimeError(f"STOP: expected {expected_total} total prompts, got {len(flat)}")

    return bank


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  Building attribute_mapping.json + prompt_bank.json")
    print("=" * 60)

    mapping = build_attribute_mapping()
    with open(ATTRIBUTE_MAPPING_JSON, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2)
    print(f"  attrs.mat order (as stored): {mapping['attrs_mat_order_as_stored']}")
    print(f"  task attribute order:        {mapping['task_attribute_order']}")
    print(f"  Saved: {ATTRIBUTE_MAPPING_JSON}")

    bank = build_prompt_bank()
    with open(PROMPT_BANK_JSON, "w", encoding="utf-8") as fh:
        json.dump(bank, fh, indent=2)
    n_total = sum(len(v["prompts"]) for v in bank["attributes"].values())
    print(f"  {len(bank['attributes'])} attributes x {len(bank['templates'])} templates = {n_total} prompts")
    print(f"  Saved: {PROMPT_BANK_JSON}")

    print("\n  Sample (Face):")
    for p in bank["attributes"]["Face"]["prompts"][:3]:
        print(f"    [{p['prompt_index']}] {p['sentence']}")
    print("  ...")
    print("=" * 60)


if __name__ == "__main__":
    main()
