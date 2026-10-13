from __future__ import annotations

import re
from typing import Any

from meta_gos.schema import QueryDecomposition, QueryOperationNeed, clean_list, clean_text


ACTION_WORDS = [
    "parse",
    "extract",
    "compute",
    "calculate",
    "convert",
    "generate",
    "render",
    "validate",
    "verify",
    "analyze",
    "build",
    "index",
    "retrieve",
    "search",
    "merge",
    "compare",
    "summarize",
    "classify",
    "filter",
    "normalize",
    "plan",
    "execute",
]
FILE_TYPES = {"csv", "docx", "html", "json", "md", "pdf", "png", "pptx", "py", "tsv", "txt", "xlsx", "xml", "yaml"}


def parse_query(task: Any) -> QueryDecomposition:
    raw_query = clean_text(task)
    lowered = raw_query.lower()
    actions = [action for action in ACTION_WORDS if re.search(rf"\b{re.escape(action)}\b", lowered)]
    if not actions:
        actions = ["perform"]
    inputs = [
        f"{kind} file"
        for kind in sorted(FILE_TYPES)
        if re.search(rf"(?:\.|\b){re.escape(kind)}\b", lowered)
    ]
    outputs = _expected_outputs(lowered)
    operations = [
        QueryOperationNeed(
            name=f"{action} {_operation_object(lowered, action)}".strip(),
            action=action,
            object=_operation_object(lowered, action),
            inputs=inputs,
            outputs=outputs,
            priority=priority,
        )
        for priority, action in enumerate(actions[:5], start=1)
    ]
    constraints = []
    if "preserve" in lowered:
        constraints.append("preserve requested existing structure or data")
    if "test" in lowered or "verify" in lowered or "validate" in lowered:
        constraints.append("validate against tests or verifier")
    return QueryDecomposition(
        raw_query=raw_query,
        normalized_goal=raw_query,
        task_type="",
        domain_objects=clean_list([item.removesuffix(" file") for item in inputs] + [op.object for op in operations]),
        input_artifacts=inputs,
        expected_outputs=outputs,
        required_operations=operations,
        constraints=constraints,
        validation_signals=constraints,
    )


def _expected_outputs(lowered_query: str) -> list[str]:
    outputs = []
    for pattern in [
        r"(?:output|produce|generate|create|return|write)\s+(?:a|an|the)?\s*([a-z0-9_\- ]{3,50})",
        r"(?:into|to)\s+(json|csv|xlsx|pdf|html|markdown|report|chart|image)",
    ]:
        for match in re.findall(pattern, lowered_query):
            outputs.append(clean_text(match))
    return clean_list(outputs)


def _operation_object(lowered_query: str, action: str) -> str:
    match = re.search(rf"\b{re.escape(action)}\b\s+(?:a|an|the)?\s*([a-z0-9_\-\. ]{{3,60}})", lowered_query)
    if match:
        phrase = re.split(r"\b(?:and|then|with|using|into|to|from|for)\b", match.group(1))[0]
        return clean_text(phrase)
    return clean_text(lowered_query[:80])
