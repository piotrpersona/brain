#!/usr/bin/env python3
import sys
import json
import pathlib
import argparse

from raganything import parser

def main():
    parser = argparse.ArgumentParser(
        prog='expand',
        description='expand brain with new memory',
    )
    parser.add_argument('memory', type=str, help='memory type')
    parser.add_argument('uri', type=str, help='URI of the memory to expand brain with')
    parser.add_argument('title', type=str, help='Title of the memory to expand brain with')
    parser.add_argument('--description', type=str, help='description of the memory to expand brain with')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    
    base_dir = pathlib.Path(__file__).resolve().parent.parent
    memory_path = base_dir / 'memory' / args.memory / 'memory.jsonl'
    
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    
    existing_uris = set()
    
    if memory_path.exists():
        with open(memory_path, 'r', encoding='utf-8') as memory_file:
            for line in memory_file:
                if line.strip():
                    try:
                        memory_object = json.loads(line)
                        existing_uris.add(memory_object.get("uri"))
                    except json.JSONDecodeError:
                        continue

    if args.uri in existing_uris:
        print(f"Memory '{args.uri}' already exists in '{args.memory}'")
        return

    new_memory_object = {
        "uri": args.uri,
        "title": args.title,
        "description": args.description if args.description else "",
    }
    
    with open(memory_path, 'a', encoding='utf-8') as memory_file:
        json.dump(new_memory_object, memory_file)
        memory_file.write('\n')
        print(f"Successfully added memory: {args.uri}")

if __name__ == '__main__':
    main()