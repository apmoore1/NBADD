import argparse
import csv
import json
from typing import List, Tuple
from pathlib import Path

import requests

def get_data(csv_url: str) -> List[Tuple[str, str]]:
  '''
  Downloads the data from the given URL address and returns it as 
  a List of Tuples where each item in the List is a sample 
  containing [text, label].
  '''
  response = requests.get(csv_url)
  response.raise_for_status()
  assert response.encoding == 'utf-8', 'Should be UTF-8 encoding'
  
  csv_lines = response.text.split('\n')
  csv_reader = csv.reader(csv_lines)
  
  data: List[Tuple[str, str]] = []
  for line_index, csv_line in enumerate(csv_reader):
    if line_index == 0 or not csv_line:
      continue
    assert len(csv_line) == 3, f'{csv_line} {line_index}'
    data_index, label, text = csv_line
    data.append((text, label))
  return data

def remove_no_text_samples(aoc_data: List[Tuple[str, str]]
                           ) -> List[Tuple[str, str]]:
  '''
  Returns the same list of data given but without samples that have no text.
  '''
  processed_data = []
  for text, label in aoc_data:
    if text.strip():
      processed_data.append((text, label))
  return processed_data

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def url_save_path(url: str, save_path: Path) -> None:
    texts_labels = remove_no_text_samples(get_data(url))
    with save_path.open('w+', encoding='utf-8', newline='') as json_file:
        for index, text_label in enumerate(texts_labels):
          text, label = text_label
          if index != 0:
            json_file.write('\n')
          json_file.write(json.dumps({'text': text, 'label': label}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder", type=parse_path,
                        help='Directory to store the data')
    args = parser.parse_args()
    data_folder: Path = args.data_folder
    data_folder.mkdir(parents=True, exist_ok=True)

    train_data_url = "https://github.com/UBC-NLP/aoc_id/raw/master/data/train/"\
                     "MultiTrain.Shuffled.csv"
    val_data_url = "https://github.com/UBC-NLP/aoc_id/raw/master/data/dev/"\
                    "MultiDev.csv"
    test_data_url = "https://github.com/UBC-NLP/aoc_id/raw/master/data/test/"\
                    "MultiTest.csv"
    urls_names = [(train_data_url, Path(data_folder, 'train.json')), 
                  (val_data_url, Path(data_folder, 'val.json')),
                  (test_data_url, Path(data_folder, 'test.json'))]
    for url_name in urls_names:
        url_save_path(*url_name)
        