# NLP Test scripts

extractor.py is the main script. It uses torotoki's (https://bitbucket.org/torotoki/corenlp-python) Python wrapper for the Java Stanford Core NLP tools to parse text, 
which needs to be set up as one (or more) JSON server(s) on `http://localhost:3456` (and ports that follow - `3457`, `3458`... - for more servers)

To specify the number of servers to be used, set the `N_SERVERS` constant in `extractor.py`.