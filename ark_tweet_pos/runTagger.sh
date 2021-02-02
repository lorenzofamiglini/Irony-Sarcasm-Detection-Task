#!/bin/bash
set -eu

# Run the tagger (and tokenizer).
java -XX:ParallelGCThreads=2 -Xmx500m -jar ark-tweet-nlp-0.3.2.jar "$@"
