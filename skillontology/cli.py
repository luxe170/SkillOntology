from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path

from skillontology.benchmark import Task, load_benchmark
from skillontology.construction.build import build_metaop_workspace
from skillontology.graph.compiler import TaskCompiler
from skillontology.retrieval.index import MetaOpVectorIndex
from skillontology.retrieval.openai_embedding import build_embedding_service
from skillontology.runtime.chat import ChatModel
from skillontology.runtime.runner import Runner


def repository_root():
    return Path(__file__).resolve().parent.parent


def load_config(path):
    config = json.loads(path.read_text(encoding='utf-8'))
    if config['benchmark'] not in ('alfworld','webshop','scienceworld'):
        raise ValueError('Unknown benchmark in configuration')
    return config


def load_tasks(path):
    payload = json.loads(path.read_text(encoding='utf-8'))
    records = payload['records']
    tasks = [Task(row['instruction'], {key:value for key,value in row.items() if key != 'instruction'}) for row in records]
    identities = [json.dumps(task.identity,sort_keys=True) for task in tasks]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate test tasks')
    return tasks


async def execute(args):
    root = repository_root()
    if args.command == 'build':
        config = load_config(args.config)
        construction = json.loads((root/'configs/construction.json').read_text(encoding='utf-8'))
        options = construction['benchmarks'][config['benchmark']]['build_config']
        embedding = None if args.agent_mode == 'deterministic' else build_embedding_service(config.get('embedding'))
        llm = None if args.agent_mode == 'deterministic' else ChatModel(config)
        memory = await build_metaop_workspace(root/config['paths']['skills'], args.output,
                                              domain_ontology=root/config['paths']['ontology'],
                                              llm_service=llm, embedding_service=embedding,
                                              agent_mode=args.agent_mode, use_llm=args.agent_mode != 'deterministic',
                                              candidate_top_k=int(options['candidate_top_k']),
                                              agent_workers=int(options['agent_workers']),
                                              snippet_chars=int(options['snippet_chars']),
                                              use_full_markdown=bool(options['use_full_markdown']),
                                              max_conflict_rounds=int(options['max_conflict_rounds_requested']) if args.agent_mode=='full' else 0)
        return {'skills':len(memory.skill_records),'metaops':len(memory.metaops),'edges':len(memory.metaop_edges)}
    config = load_config(args.config)
    tasks = load_tasks(root/config['paths']['testset'])
    index = MetaOpVectorIndex.load(root/config['paths']['assets'])
    if args.command == 'check':
        load_benchmark(config['benchmark'], config['environment'])
        return {'benchmark':config['benchmark'],'tasks':len(tasks),'metaops':len(index.metaops),'embedding_dimension':index.vectors.shape[1]}
    task = tasks[args.task_index]
    adapter, environment = load_benchmark(config['benchmark'], config['environment'])
    compiler = TaskCompiler(root/config['paths']['assets'], build_embedding_service(config['embedding']), config['retrieval'])
    result = await Runner(adapter, environment, compiler, ChatModel(config), config).run(task)
    return asdict(result)


def main():
    parser = argparse.ArgumentParser(prog='skillontology')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('check','run','build'):
        command = commands.add_parser(name)
        command.add_argument('--config', type=Path, required=True)
        if name == 'run':
            command.add_argument('--task-index', type=int, required=True)
        if name == 'build':
            command.add_argument('--output', type=Path, required=True)
            command.add_argument('--agent-mode', choices=('deterministic','extract_only','full'), default='full')
    args = parser.parse_args()
    print(json.dumps(asyncio.run(execute(args)),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
