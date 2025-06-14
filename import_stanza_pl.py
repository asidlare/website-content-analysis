import stanza

# Download Polish language package
stanza.download(
    'pl',
    processors='tokenize,mwt,pos,lemma'
)
