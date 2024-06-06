from retrieve_cat import NAME
from retrieve_cat.rag.rag_config import RagConfig
def build_index(parser):
    parser.add_argument("-s", "--src", required=True, help="src documents")
    parser.add_argument("-c", "--config", required=True, help="config path")
    args = parser.parse_args()

    config = RagConfig.from_file(args.config)

    collection = config.collection 
    collection.ingest(args.src, config.chunk_size, config.chunk_overlap)

    # with open(f"rag.sh", "w") as f:
    #     f.write(f"{NAME} rag -e chromadb -m {args.model} -f {args.dest} -q \"$1\"")
