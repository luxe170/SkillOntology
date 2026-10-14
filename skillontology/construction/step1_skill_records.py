from __future__ import annotations
from skillontology.schema import SkillNode
import asyncio
from pathlib import Path
import json
import re
from types import SimpleNamespace
from typing import Any, Iterable
import yaml
from skillontology.schema import SkillRecord

def _clean_frontmatter_value(value: Any) -> str:
    if isinstance(value, list):
        return ', '.join((str(item) for item in value if str(item).strip()))
    return str(value or '').strip()

def _clean_frontmatter_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    else:
        raw = re.split('[\\n;,]+', str(value))
    return [str(item).strip() for item in raw if str(item).strip()]

def _fallback_skill_node_from_content(path: Path, content: str, *, snippet_chars: int) -> SkillNode:
    metadata: dict[str, Any] = {}
    body = content
    if content.strip().startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            parsed = yaml.safe_load(parts[1]) or {}
            if isinstance(parsed, dict):
                metadata = parsed
                body = parts[2]
    allowed_tools = _clean_frontmatter_list(metadata.get('allowed_tools') or metadata.get('allowed-tools') or metadata.get('tools'))
    if not allowed_tools:
        match = re.search('(?im)^\\s*(?:#+\\s*)?Allowed tools\\s*:?\\s*(.+)$', content)
        if match:
            allowed_tools = _clean_frontmatter_list(match.group(1))
    return SkillNode.from_lists(name=_clean_frontmatter_value(metadata.get('name')) or path.parent.name, description=_clean_frontmatter_value(metadata.get('description')), one_line_capability=_clean_frontmatter_value(metadata.get('one_line_capability') or metadata.get('capability')), inputs=_clean_frontmatter_list(metadata.get('inputs')), outputs=_clean_frontmatter_list(metadata.get('outputs')), domain_tags=_clean_frontmatter_list(metadata.get('domain_tags') or metadata.get('domains')), tooling=_clean_frontmatter_list(metadata.get('tooling')), example_tasks=_clean_frontmatter_list(metadata.get('example_tasks')), script_entrypoints=_clean_frontmatter_list(metadata.get('script_entrypoints')), compatibility=_clean_frontmatter_list(metadata.get('compatibility')), allowed_tools=allowed_tools, source_path=str(path), rendered_snippet=body.strip()[:snippet_chars], raw_content=content, metadata=metadata, skill_id=_clean_frontmatter_value(metadata.get('skill_id')) or str(path))

def iter_skill_md_files(skills_dir: str | Path) -> Iterable[Path]:
    base = Path(skills_dir).expanduser()
    if base.is_file() and base.name == 'SKILL.md':
        yield base
        return
    for path in sorted(base.rglob('SKILL.md')):
        if path.is_file():
            yield path

def load_skill_records(skills_dir: str | Path, *, snippet_chars: int=900) -> list[SkillRecord]:
    records: list[SkillRecord] = []
    for path in iter_skill_md_files(skills_dir):
        node = skill_node_from_skill_md(path, snippet_chars=snippet_chars)
        if node is not None:
            records.append(SkillRecord.from_skill_node(node))
    return records

async def async_load_skill_records(skills_dir: str | Path, llm_service: Any | None, *, snippet_chars: int=900, use_full_markdown: bool=False, workers: int=1) -> list[SkillRecord]:
    paths = list(iter_skill_md_files(skills_dir))
    if not paths:
        return []
    worker_count = max(1, int(workers or 1))
    if worker_count == 1 or len(paths) <= 1:
        records: list[SkillRecord] = []
        for path in paths:
            node = await async_skill_node_from_skill_md(path, llm_service, snippet_chars=snippet_chars, use_full_markdown=use_full_markdown)
            if node is not None:
                records.append(SkillRecord.from_skill_node(node))
        return records
    semaphore = asyncio.Semaphore(worker_count)

    async def run_one(index: int, path: Path) -> tuple[int, SkillRecord | None]:
        async with semaphore:
            node = await async_skill_node_from_skill_md(path, llm_service, snippet_chars=snippet_chars, use_full_markdown=use_full_markdown)
            return (index, SkillRecord.from_skill_node(node) if node is not None else None)
    tasks = [asyncio.create_task(run_one(index, path)) for index, path in enumerate(paths)]
    ordered: list[SkillRecord | None] = [None for _ in paths]
    for future in asyncio.as_completed(tasks):
        index, record = await future
        ordered[index] = record
    return [record for record in ordered if record is not None]

def save_skill_records(records: list[SkillRecord], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(record.to_json() + '\n')

def load_skill_records_jsonl(path: str | Path) -> list[SkillRecord]:
    records: list[SkillRecord] = []
    with Path(path).open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                records.append(SkillRecord(**json.loads(line)))
    return records

def skill_node_from_skill_md(path, *, snippet_chars=900):
    return _fallback_skill_node_from_content(path, path.read_text(encoding='utf-8'), snippet_chars=snippet_chars)

async def async_skill_node_from_skill_md(path, llm_service, *, snippet_chars=900, use_full_markdown=False):
    return skill_node_from_skill_md(path, snippet_chars=snippet_chars)
