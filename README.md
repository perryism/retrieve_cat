# Init

The command below will bootstrap a config file for the application

<pre>
python -m retrieve_cat.cli.main init -n benefit
</pre>

# Index

Provide a text file and the config created from the previous step, we will create a vector store and store the embeddings.

<pre>
python -m retrieve_cat.cli.main index -s data/benefits.txt -c benefit.yaml
</pre>

# Chat

We can now chat with the bot about the text digested from the previous step.

<pre>
python -m retrieve_cat.cli.main chat -c benefit.yaml
</pre>
