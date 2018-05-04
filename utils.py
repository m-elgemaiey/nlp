import json
import spacy

'''
Load the intents info from the training file
'''
def load_intents(file_name):
    sentences = []
    intents = []
    intent_set = set()

    with open(file_name) as json_file:
        json_data = json.load(json_file)
        examples = json_data['rasa_nlu_data']['common_examples']
        for example in examples:
            sentences.append(example['text'])
            intent = example['intent']
            intents.append(intent)
            intent_set.add(intent)

    return sentences, intents, intent_set

'''
Load the entities from the training file
'''
def load_entities(file_name):
    entity_set = set()
    with open(file_name) as json_file:
        json_data = json.load(json_file)
        examples = json_data['rasa_nlu_data']['common_examples']
        for example in examples:
            entities = example['entities']
            if entities:
                for entity in entities:
                    entity_set.add(entity['entity'])

    return examples, entity_set

def load_spacy():
    print("Loading spacy model...")
    nlp = spacy.load('en_core_web_md')
    print("Loaded spacy model")
    return nlp
