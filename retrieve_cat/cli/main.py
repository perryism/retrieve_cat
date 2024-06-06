import argparse
import logging
import sys
from .commands.vectorstore import build_index
from .commands.rag import build_rag
from .commands.init import init
from .commands.chat import chat

from retrieve_cat import NAME

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(prog = NAME)

mapping = {
    "init": init,
    "index": build_index,
    "rag": build_rag,
    "chat": chat
}

parser.add_argument("cmd", help="command", choices=mapping.keys())

def main():
    global parser
    args = parser.parse_known_args()
    func = mapping[args[0].cmd]
    func(parser)

if __name__ == "__main__":
    main()