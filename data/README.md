# Data

Training text files live here (they are git-ignored).

The arithmetic and shapes demos generate their own data, so nothing is needed
for them. To train the character-level GPT on real text, drop any `.txt` file
here. A classic choice is the Tiny Shakespeare corpus (about 1 MB):

```bash
curl -L -o data/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Then run:

```bash
python examples/train_text_gpt.py --data data/shakespeare.txt --dashboard
```
