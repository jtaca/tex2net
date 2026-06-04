import re

import pytest


class FakeEntity:
    def __init__(self, text, label_="PERSON"):
        self.text = text
        self.label_ = label_


class FakeHead:
    def __init__(self, text="", dep_=""):
        self.text = text
        self.dep_ = dep_


class FakeToken:
    def __init__(self, text, dep_="", pos_="", lemma_=None, head=None):
        self.text = text
        self.dep_ = dep_
        self.pos_ = pos_
        self.lemma_ = lemma_ if lemma_ is not None else text.lower()
        self.head = head or FakeHead()


class FakeSentence:
    def __init__(self, text):
        self.text = text.strip()
        names = re.findall(r"\b[A-Z][a-z]+\b", self.text)
        self.ents = [FakeEntity(name) for name in dict.fromkeys(names)]

        raw_tokens = re.findall(r"[A-Za-z']+", self.text)
        root_index = None
        for index, token in enumerate(raw_tokens):
            if token.isalpha() and token[0].islower():
                root_index = index
                break

        self._tokens = []
        for index, token in enumerate(raw_tokens):
            is_root = index == root_index
            dep = "ROOT" if is_root else ""
            pos = "VERB" if is_root else ("PROPN" if token[:1].isupper() else "")
            self._tokens.append(FakeToken(token, dep_=dep, pos_=pos, lemma_=token.lower()))

        root_token = next((token for token in self._tokens if token.dep_ == "ROOT"), FakeToken("interacts", dep_="ROOT", pos_="VERB"))
        for token in self._tokens:
            token.head = root_token

    def __iter__(self):
        return iter(self._tokens)


class FakeDoc:
    def __init__(self, text):
        sentence_texts = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text.strip()) if segment.strip()]
        self.sents = [FakeSentence(sentence_text) for sentence_text in sentence_texts] or [FakeSentence(text)]
        self.ents = []
        for sentence in self.sents:
            self.ents.extend(sentence.ents)

    def __getitem__(self, item):
        return self


class FakeNLP:
    pipe_names = ("sentencizer",)

    def __call__(self, text):
        return FakeDoc(text)


@pytest.fixture
def fake_nlp():
    return FakeNLP()