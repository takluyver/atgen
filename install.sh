!/bin/bash

# The PyPI version submodlib seems to be broken, so
# one has to resort to manual installation.
echo "Installing submodlib"
git clone https://github.com/decile-team/submodlib.git
cd submodlib
# Cannot run `pip install -r requirements.txt` due to versions mismatch
pip install sphinxcontrib-bibtex pybind11>=2.6.0 scikit-learn scipy
pip install . --no-deps
cd ..
rm -rf submodlib

echo "Installing AlignScore..."
# Install requirements for AlignScore
pip install summac==0.0.3 --no-deps
pip install src/atgen/metrics/AlignScore --no-deps
python -m spacy download en_core_web_sm
echo "Installing deps for al-nlg..."
python3 -c "import nltk ; nltk.download('punkt')"

cd src/atgen/metrics/AlignScore/ && wget https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt
echo "Done!"
