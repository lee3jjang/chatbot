import numpy as np
import random
import nltk
# nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

from rich.console import Console
from rich.traceback import install
from rich.progress import track
install()
console = Console()


# Tokenize
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Stem
def stem(word):
    return stemmer.stem(word.lower())

# Bag of Words
def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag


# Test
if __name__ == '__main__':
    
    # Test 1
    console.log(tokenize("Hello, World!"))

    # Test 2
    words = [stem(w) for w in ["organize", "organizes", "organizing"]]
    console.log(words)

    # Test 3
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag = bag_of_words(sentence, words)
    console.log(bag)