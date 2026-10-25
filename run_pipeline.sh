#!/bin/sh

make data
make tts
make select
make eval ANSWERER=A JUDGE=B
make variance ANSWERER=A JUDGE=B
make report
