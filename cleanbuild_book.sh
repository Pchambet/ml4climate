rm -f book/bibtex.json
jupyter-book clean -a book
jupyter-book build -v book
