from shutil import rmtree
import streamlit as st
from datasets import load_from_disk
from time import sleep

from atgen.utils.get_last_workdir import get_last_workdir


iter_number = 0

workdir = get_last_workdir()
path = workdir / 'dataset_to_annotate'

def reset_session_state():
    st.session_state.annotations = []
    st.session_state.current_index = 0

if path.exists():
    st.text('Kindly annotate the texts. Please do not reload the page!')
    sleep(1)
    dataset = load_from_disk(path)
    texts = dataset['input']
    num_texts = len(texts)
    if 'annotations' not in st.session_state:
        st.session_state.annotations = []
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0

    if st.session_state.current_index < num_texts:
        text = texts[st.session_state.current_index]
        st.write(f"Text {st.session_state.current_index + 1}/{num_texts}")
        st.write(text)

        annotation = st.text_area("Enter your annotation", key=f"annotation_{st.session_state.current_index}")
        if st.button("Submit"):
            st.session_state.annotations.append(annotation)
            st.session_state.current_index += 1
            st.rerun()

    if len(st.session_state.annotations) == num_texts:
        st.write("Annotation completed! Kindly wait for the next batch of texts to be ready for annotation. In the meantime, inspect the model metrics on the eval sample or its examples of generated texts.")
        dataset = dataset.add_column(
            'annotation', st.session_state.annotations
        )
        dataset.save_to_disk(workdir / 'annotated_query')
        rmtree(path)
        reset_session_state()

else:
    st.text('No texts are curently available for annotation. If you are an annotator, kindly wait for the texts to be selected for annotation.')