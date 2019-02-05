from typing import List, Tuple
import json
from pathlib import Path

from sklearn.metrics import f1_score, accuracy_score

SAVE_DIR = Path('.', 'confusion_matrix_results')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PATHS_NAMES = [('No Attention Model', Path(SAVE_DIR, 'No Attention Model.json')),
               ('Standard Model', Path(SAVE_DIR, 'Standard Model.json')),
               ('Code Switching', Path(SAVE_DIR, 'Code Switching.json')),
               ('Bivalency', Path(SAVE_DIR, 'Bivalency.json')),
               ('Strong Bivalency', Path(SAVE_DIR, 'Strong Bivalency.json')),
               ('Bivalency and Code Switching', Path(SAVE_DIR, 'Bivalency and Code Switching.json'))]

def load_scores(score_path: Path) -> Tuple[List[str], List[str]]:
    true_scores = []
    pred_scores = []
    with score_path.open('r') as scores:
        for score in scores:
            score = json.loads(score)
            pred_scores.append(score['prediction'])
            true_scores.append(score['label'])
    return true_scores, pred_scores


for name, model_path in PATHS_NAMES:
    true_score, pred_score = load_scores(model_path)
    print(name)
    print(f'Accuracy {accuracy_score(true_score, pred_score)}')
    print(f'Macro F1 {f1_score(true_score, pred_score, average="macro")}')
    print('----------------')


    