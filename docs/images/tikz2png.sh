#!/bin/sh

pdflatex $1.tex
sips -s format png $1.pdf --out $1.png
rm $1.aux $1.log $1.pdf 
