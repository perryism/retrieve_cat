from retrieve_cat.rag.utils import Rag
from retrieve_cat.rag.rag_config import RagConfig

def build_rag(parser):
    parser.add_argument("-q", "--question", required=True, help="question")
    parser.add_argument("-c", "--config", required=True, help="config path")

    args = parser.parse_args()
    config = RagConfig.from_file(args.config)
    rag = Rag(config)

    print(rag.query(args.question))
