from retrieve_cat import NAME
from retrieve_cat.rag.rag_config import RagConfig
import logging
from retrieve_cat.vector_store.collection import Collection

logger = logging.getLogger(__name__)


from langchain_core.documents import Document

def build_index(parser):
    parser.add_argument("-c", "--config", required=True, help="config path")
    parser.add_argument("-d", "--delete", action="store_true" , help="config path")
    args = parser.parse_args()

    config = RagConfig.from_file(args.config)

    collection : Collection = config.collection

    if args.delete:
        logger.info("Deleting existing collection")
        collection.delete()

    for source in config.sources:
        logger.info(f"ingesting {source}")
        collection.ingest(source, config.chunk_size, config.chunk_overlap)

    # with open(f"rag.sh", "w") as f:
    #     f.write(f"{NAME} rag -e chromadb -m {args.model} -f {args.dest} -q \"$1\"")
