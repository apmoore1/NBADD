import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import List, Dict


CLASS_LABELS = ['gulf', 'egyptian', 'msa', 'levantine']

def folder_to_json(class_folder_fp: Path, verbose: bool
                   ) -> List[Dict[str, str]]:
    class_label = class_folder_fp.name
    json_data: List[Dict[str, str]] = []
    for text_fp in class_folder_fp.iterdir():
        if class_label not in text_fp.name:
            continue
        with text_fp.open('r') as text_file:
            text = text_file.read().strip()
            if text == '':
                raise ValueError(f'The file {text_fp} should contain some text')
            json_data.append({'text': text, 'label': class_label})
    if verbose:
        print(f'Class Label {class_label}, number of samples: '
              f'{len(json_data)}')
    return json_data

def folders_to_json(folder_fp: Path, json_fp: Path, verbose: bool) -> None:
    class_folder_paths: List[Path] = []
    for class_folder_path in folder_fp.iterdir():
        if class_folder_path.name in CLASS_LABELS:
            class_folder_paths.append(class_folder_path)
    assert len(class_folder_paths) == len(CLASS_LABELS)
    all_json_data: List[Dict[str, str]] = []
    for class_folder_path in class_folder_paths:
        all_json_data.extend(folder_to_json(class_folder_path, verbose))
    
    if verbose:
        label_stats = defaultdict(lambda: 0)
        for label_text in all_json_data:
            label_stats[label_text['label']] += 1
        print(label_stats)
    with json_fp.open('w', encoding='utf-8') as json_file:
        json.dump(all_json_data, json_file, ensure_ascii=False)

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("data_dir", type=parse_path,
                        help='Directory that contains the class label folders')
    parser.add_argument("json_file", help='File to dump the json data to', 
                        type=parse_path)
    args = parser.parse_args()
    data_dir = args.data_dir
    json_fp = args.json_file
    folders_to_json(data_dir, json_fp, verbose=args.verbose)
