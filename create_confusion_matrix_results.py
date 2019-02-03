import json
from pathlib import Path

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

import nbadd

TEST_FILE = Path('.', 'clean_aoc_data', 'test.json')
TEST_DATA = []
with TEST_FILE.open('r', encoding='utf-8') as json_data:
    for line in json_data:
        TEST_DATA.append(json.loads(line))

MODELS_DIR = Path('.', 'models')
SAVE_DIR = Path('.', 'confusion_matrix_results')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PATHS_NAMES = [('Standard Model', Path(MODELS_DIR, 'word_char_attention')),
               ('Code Switching', Path(MODELS_DIR, 'word_char_attention_code_switch')),
               ('Bivalency', Path(MODELS_DIR, 'word_char_attention_bivalency')),
               ('Bivalency and Code Switching', Path(MODELS_DIR, 'word_char_attention_bivalency_code'))]


for name, model_path in PATHS_NAMES:
    model_path = Path(model_path, 'model.tar.gz')
    archive = load_archive(model_path.resolve())
    predictor = Predictor.from_archive(archive, 'dialect-predictor')
    save_file = Path(SAVE_DIR, f'{name}.json')
    with save_file.open('w+') as save_data:
        for index, data_to_predict in enumerate(TEST_DATA):
            test_label = data_to_predict['label']
            prediction = predictor.predict_json({'text': data_to_predict['text']})
            if index != 0:
                save_data.write('\n')
            save_data.write(json.dumps({'prediction': prediction['label'], 
                                        'label': test_label}))


    