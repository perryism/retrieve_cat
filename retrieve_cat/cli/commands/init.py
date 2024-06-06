import yaml

def init(parser):
    parser.add_argument("-n", "--name", required=True, help="name")
    args = parser.parse_args()

    config = {
        "name": args.name, 
        "embedding_model": "all-MiniLM-L6-v2",
        "persistent_path": f"./db/{args.name}",
        "index": {
            "engine": "chromadb",
            "chunk_size": 200,
            "chunk_overlap": 50
        },
        "llm": {
            "type": "local",
            "args": {
                "base_url": "http://localhost:1234/v1"
            }
        }
    }

    with open(f"{args.name}.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)