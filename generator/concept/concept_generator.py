import spacy
import random
import tensorflow.compat.v1 as tf

class ConceptGenerator:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.pipeline = [("tagger", self.nlp.tagger), ("parser", self.nlp.parser)]

    #TODO : can generate concept shuffling ????
    def check_availability(self, sentence):
        def check_availability_sentence(x):
            x = x.numpy().decode('utf-8')
            doc = self.nlp(str(x))
            V_concepts = []
            N_concepts = []
            original_tokens = []
            for token in doc:
                original_tokens.append(token.text_with_ws)
                if token.pos_.startswith('V') and token.is_alpha and not token.is_stop:
                    V_concepts.append(token.text_with_ws)
            for noun_chunk in doc.noun_chunks:
                root_noun = noun_chunk[-1]
                if root_noun.pos_ == "NOUN":
                    N_concepts.append(root_noun.text_with_ws)
            if len(N_concepts) >= 2 or len(V_concepts) >= 2:
                return True
            else:
                return False

        result = tf.py_function(check_availability_sentence, [sentence['text']], [tf.bool])[0]
        return result

    def generate(self, prompt):
        doc = self.nlp(str(prompt))
        V_concepts = []
        N_concepts = []
        original_tokens = []
        for token in doc:
            original_tokens.append(token.text_with_ws)
            if token.pos_.startswith('V') and token.is_alpha and not token.is_stop:
                V_concepts.append(token.text_with_ws)
        for noun_chunk in doc.noun_chunks:
            root_noun = noun_chunk[-1]
            if root_noun.pos_ == "NOUN":
                N_concepts.append(root_noun.text_with_ws)

        if len(N_concepts) >= 2:
            random.shuffle(N_concepts)
        if len(V_concepts) >= 2:
            random.shuffle(V_concepts)

        shuffled_tokens = []
        N_concepts_index = 0
        V_concepts_index = 0
        for tok in original_tokens:
            if tok in V_concepts and V_concepts_index < len(V_concepts):
                shuffled_tokens.append(V_concepts[V_concepts_index])
                V_concepts_index += 1
            elif tok in N_concepts and N_concepts_index < len(N_concepts):
                shuffled_tokens.append(N_concepts[N_concepts_index])
                N_concepts_index += 1
            else:
                shuffled_tokens.append(tok)

        assert len(shuffled_tokens) == len(original_tokens)

        result = ''.join([token for token in shuffled_tokens])
        return result