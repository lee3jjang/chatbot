from chatbot.nltk_utils import bag_of_words, tokenize, stem
import numpy as np

def test_tokenize():
    token = tokenize("Hello, World!")
    assert token == ['Hello', ',', 'World', '!']

def test_stem():
    words = [stem(w) for w in ["organize", "organizes", "organizing"]]
    assert words == ['organ', 'organ', 'organ']

def test_bag_of_words():
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag = bag_of_words(sentence, words)
    assert np.array_equal(bag, np.array([0., 1., 0., 1., 0., 0., 0.]))