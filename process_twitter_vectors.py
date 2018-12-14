import argparse
from pathlib import Path

from gensim.models import Word2Vec
from tqdm import tqdm

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    vector_file_path_help = ("File Path to the Twitter Vectors that have been "
                             "downloaded and are currently in a Gensim format")
    output_vector_help = ("File path to save the vector data in a line by "
                          "line format.")

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("vector_file_path", type=parse_path, 
                        help=vector_file_path_help)
    parser.add_argument("--output_vector_file_path", type=parse_path, 
                        help=output_vector_help, default="./TwitterVectors.txt")
    args = parser.parse_args()
    
    vector_fp: Path = args.vector_file_path
    output_vector_fp: Path = args.output_vector_file_path
    if not vector_fp.exists():
        raise ValueError(f'The vector file path given {vector_fp} '
                         'does not exist')
    
    count = 0
    word_vector = Word2Vec.load(str(vector_fp.resolve()))
    with output_vector_fp.open('w') as vector_file:
        for index, word in enumerate(tqdm(word_vector.wv.vocab.keys())):
            vector = word_vector[word]
            string_vector = [str(v) for v in vector]
            string_vector.insert(0, word)
            string_vector = ' '.join(string_vector)
            if index != 0:
                string_vector = '\n' + string_vector
            vector_file.write(string_vector)
            count += 1
    if args.verbose:
        print(f'Number of word in word vector file: {count}')
    if count != 202690:
        raise ValueError('The number of words in the Word Vector file should '
                         f'be 202690 and not {count}')