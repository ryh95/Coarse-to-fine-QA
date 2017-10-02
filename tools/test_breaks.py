import json

with open('../data/validation-0.json', 'r') as js_file:
    for idx, sample in enumerate(js_file):
        dict_sample = json.loads(sample)
        # use docuement vocab
        # answer = dict_sample['answer_sequence']
        # question = dict_sample['question_sequence']
        document = dict_sample['document_sequence']
        string_sequences = dict_sample['string_sequence']
        sentence_breaks = dict_sample['sentence_breaks']
        paragraph_breaks = dict_sample['paragraph_breaks']

        breaks = sorted(sentence_breaks)
        breaks = [0] + breaks + [len(string_sequences)]

        assert len(document) == len(string_sequences)

        sentences = []
        for i, id_break in enumerate(breaks[:-1]):
            start = breaks[i]
            end = breaks[i+1]
            sentences.append(string_sequences[start:end])