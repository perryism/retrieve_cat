from retrieve_cat.rag.rag_config import RagConfig
from retrieve_cat.rag.utils import Rag
import warnings
warnings.filterwarnings("ignore")

def chat(parser):
    parser.add_argument("-c", "--config", required=True, help="config path")
    args = parser.parse_args()

    config = RagConfig.from_file(args.config)
    rag = Rag(config)

    while True:
        query = input("You: ")
        result = rag.query(query)
        if result:
            print("Bot: ",  result)
        else:
            print("Bot: I didn't find any answers")
