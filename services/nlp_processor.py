import difflib
import re
from dataclasses import dataclass
from functools import lru_cache

from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

SPELLING_CUTOFF = 0.84
MIN_CORRECTABLE_LENGTH = 5

_TOKENIZER = RegexpTokenizer(r"[a-z0-9+#]+")
_STEMMER = PorterStemmer()

STOPWORDS = frozenset(
    """a an the i me my we our you your is am are was were be been being to of in on at for with and or but
    so as by from into about that this these those it its im ive want wants like love enjoy interest interested
    interests would could should can will do does did have has had get got work working job jobs career careers
    field also very really some any please tell show give help need looking look good best what which who how
    when where why more most much many than then there here just not no yes etc become people""".split()
)

PHRASE_ALIASES = (
    (r"\bcyber[\s-]?security\b", "cybersecurity"),
    (r"\bcyber\b", "cybersecurity"),
    (r"\bhackers?\b", "hacking"),
    (r"\bpen[\s-]?test(ing)?\b", "penetration testing"),
    (r"\bai\b", "artificial intelligence"),
    (r"\bml\b", "machine learning"),
    (r"\bnlp\b", "natural language processing"),
    (r"\bscikit[\s-]?learn\b", "scikit learn"),
    (r"\bnode[\s.]?js\b", "node javascript"),
    (r"\bjs\b", "javascript"),
    (r"\bci\s*/\s*cd\b", "ci cd"),
    (r"\bfront[\s-]?end\b", "frontend"),
    (r"\bback[\s-]?end\b", "backend"),
    (r"\bwebsites?\b", r"\g<0> web"),
    (r"\bamazon web services\b", "aws"),
    (r"\b(gcp|google cloud)\b", "cloud"),
    (r"\bprogrammers?\b", "programming"),
    (r"\bcoders?\b", "coding"),
    (r"\bdesigners?\b", "design"),
)
_COMPILED_ALIASES = tuple((re.compile(pattern), replacement) for pattern, replacement in PHRASE_ALIASES)


@dataclass(frozen=True)
class ProcessedText:
    original: str
    tokens: tuple
    stems: frozenset

    @property
    def is_empty(self):
        return not self.stems


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    normalized = text.lower().replace("&", " and ")
    for pattern, replacement in _COMPILED_ALIASES:
        normalized = pattern.sub(replacement, normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize(text):
    return _TOKENIZER.tokenize(normalize_text(text))


def remove_stopwords(tokens):
    return [token for token in tokens if token not in STOPWORDS]


def stem_tokens(tokens):
    return [_STEMMER.stem(token) for token in tokens]


def build_vocabulary(phrases):
    words = set()
    for phrase in phrases:
        words.update(token for token in tokenize(phrase) if token.isalpha() and token not in STOPWORDS)
    return frozenset(words)


def correct_spelling(tokens, vocabulary):
    if not vocabulary:
        return list(tokens)
    candidates = sorted(vocabulary)
    corrected = []
    for token in tokens:
        if len(token) >= MIN_CORRECTABLE_LENGTH and token.isalpha() and token not in vocabulary:
            close_matches = difflib.get_close_matches(token, candidates, n=1, cutoff=SPELLING_CUTOFF)
            corrected.append(close_matches[0] if close_matches else token)
        else:
            corrected.append(token)
    return corrected


def process_text(text, vocabulary=frozenset()):
    original = text if isinstance(text, str) else ""
    tokens = remove_stopwords(tokenize(original))
    tokens = correct_spelling(tokens, vocabulary)
    return ProcessedText(original=original, tokens=tuple(tokens), stems=frozenset(stem_tokens(tokens)))


@lru_cache(maxsize=None)
def phrase_stems(phrase):
    return frozenset(stem_tokens(remove_stopwords(tokenize(phrase))))


def phrase_matches(phrase, stems):
    required = phrase_stems(phrase)
    return bool(required) and required <= stems


def extract_keywords(processed_text, phrases):
    return [phrase for phrase in phrases if phrase_matches(phrase, processed_text.stems)]
