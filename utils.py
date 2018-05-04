import json

def load_intents(file_name):
    sentences = []
    intents = []
    intent_set = set()

    with open(file_name) as json_file:
        json_data = json.load(json_file)
        examples = json_data['rasa_nlu_data']['common_examples']
        for example in examples:
            sentences.append(example['text'])
            intents.append(example['intent'])
            intent_set.add(example['intent'])

    return sentences, intents, intent_set