# GPT Memory

Keeps a summary of everything GPT rates as sufficiently important to be worth remembering (sometimes this includes praising you or remembering how wonderful you are, it's got some pretty messed up priorities...), then during conversation will find memories related to what's currently being said so it can refer back to them.
There's also 'dialogue compression' which summarises the dialogue so far and uses that for reference in future conversation - this results in it having better long-term continuity, especially in combination with having the semantic memory.

Now has a discord front-end so you can add it to a channel, and theoretically also a slack front-end but it was written entirely by GPT and is completely untested.


## Installation
<code>git clone https://github.com/ari-bc/gpt-semantic-memory.git
cd gpt-semantic-memory
python3 -m venv venv
pip install -r requirements.txt
</code>
