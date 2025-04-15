from flask import Flask, render_template, request, redirect, url_for
import spacy
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from transformers import pipeline

app = Flask(__name__)


def transformers_result(text, option, language):
    # Map language to appropriate model
    if language != 'en':
        return ['Hugging face(Bert RoBerta) works with Russian not properly.']

    model_name = "dslim/bert-base-NER"  # English NER model

    # Create NER pipeline
    ner_pipeline = pipeline("ner", model=model_name, aggregation_strategy="simple")

    # Process text
    entities = ner_pipeline(text)
    result = []

    # Filter entities by the requested type
    for entity in entities:
        if entity['entity_group'] == option:
            result.append(entity['word'])

    return result


def nltk_result(text, option, language='en'):
    # NLTK's NER is primarily English-only (Russian support is limited)
    if language != 'en':
        return ['NLTK works with Russian not properly.']

    option_map = {
        'PER': 'PERSON',
        'ORG': 'ORGANIZATION',
        'GPE': 'GPE',  # Geo-Political Entity
        'DATE': 'DATE',
        'MONEY': 'MONEY'
    }

    nltk_label = option_map.get(option)

    # Tokenize and tag
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    # Named entity chunking
    chunks = ne_chunk(tagged)

    result = []
    for chunk in chunks:
        if isinstance(chunk, Tree):
            label = chunk.label()
            if label == nltk_label:
                entity = " ".join([token for token, pos in chunk.leaves()])
                result.append(entity)

    return result


def spacy_result(text, option, language):
    if language == 'ru':
        if option == 'GPE':
            option = 'LOC'

        nlp = spacy.load("ru_core_news_sm")
    else:
        nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    result = []

    for ent in doc.ents:
        if ent.label_ == option:
            result.append(ent.text)

    return result


@app.route('/', methods=['GET', 'POST'])
def index():
    nltk_results = {}
    spacy_results = {}
    transformer_results = {}
    text = ''
    methods_button = []

    if request.method == 'POST':
        language = request.form.get('language')
        text = request.form.get('text')
        entity_options = request.form.getlist('entities_button')
        methods_button = request.form.getlist('methods_button')

        # Handle "All" selection for entities
        if 'all' in entity_options or not entity_options:
            entity_options = ['PER', 'ORG', 'GPE', 'DATE', 'MONEY']

        # Handle "All" selection for methods
        if 'all' in methods_button or not methods_button:
            methods_button = ['spacy', 'nltk', 'transformer_results']

        # Process each selected method
        if 'spacy' in methods_button:
            spacy_results = {}
            for entity in entity_options:
                result = spacy_result(text, entity, language)
                if result:  # Only add if we found something
                    spacy_results[entity] = result

        if 'nltk' in methods_button:
            nltk_results = {}
            for entity in entity_options:
                result = nltk_result(text, entity, language)
                if result:
                    nltk_results[entity] = result

        if 'transformer_results' in methods_button:
            transformer_results = {}
            for entity in entity_options:
                if language == 'en':  # Transformer only works well with English
                    result = transformers_result(text, entity, language)
                    if result:
                        transformer_results[entity] = result
                else:
                    transformer_results[entity] = ['Hugging face(Bert RoBerta) works with Russian not properly.']

    return render_template('index_old.html',
                           spacy_results=spacy_results,
                           nltk_results=nltk_results,
                           transformer_results=transformer_results,
                           previous_text=text,
                           methods_button=methods_button)


if __name__ == "__main__":
    app.run(debug=True)