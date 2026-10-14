from dataclasses import replace

from .domain_ontology import clean_text


def ground_query(query, ontology):
    aliases = ontology.action_aliases()
    needs = []
    for need in query.required_operations:
        action = clean_text(need.action).lower()
        canonical = aliases.get(action, aliases.get(action.replace('_', ' '), action))
        needs.append(replace(need, action=canonical))
    return replace(query, required_operations=needs)
