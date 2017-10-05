#!/bin/bash

echo "Downloading WikiReading json data(vocab and validation)..."

CLOUD_STORAGE=https://storage.googleapis.com/wikireading

# wget https://github.com/google-research-datasets/wiki-reading/blob/master/README.md
wget ${CLOUD_STORAGE}/answer.vocab
wget ${CLOUD_STORAGE}/document.vocab
wget ${CLOUD_STORAGE}/raw_answer.vocab
wget ${CLOUD_STORAGE}/type.vocab

mv *.vocab ../data/vocab

wget -c ${CLOUD_STORAGE}/validation.json.tar.gz
tar xvzf validation.json.tar.gz
mv validation-*.json ../data/original_data
rm validation.json.tar.gz
# wget -c ${CLOUD_STORAGE}/test.tar.gz
# tar xvzf test.tar.gz &
# wget -c ${CLOUD_STORAGE}/train.tar.gz
# tar xvzf train.tar.gz

echo "Done."