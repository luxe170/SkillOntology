from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import re

from skillontology.runtime.llm_json import call_json


KINDS = ('object_role','state_predicate','action_alias','transition','workflow','conditional_prerequisite','recovery')


def public_training_episodes(records, excluded_identities=()):
    excluded = {json.dumps(identity,sort_keys=True) for identity in excluded_identities}
    episodes = []
    identities = set()
    for record in records:
        if record.get('split') not in ('train','official_train','memory_train'):
            raise ValueError('Ontology induction accepts training records only')
        identity = json.dumps(record['identity'],sort_keys=True)
        if identity in excluded or identity in identities:
            raise ValueError('Training/evaluation overlap or duplicate identity')
        identities.add(identity)
        trace = [{'action':str(step['action']),'observation':str(step['observation'])} for step in record['actions']]
        episode = {'instruction':str(record['instruction']),'trace':trace}
        episode['id'] = hashlib.sha256(json.dumps(episode,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()
        episodes.append(episode)
    if len({episode['id'] for episode in episodes}) != len(episodes):
        raise ValueError('Duplicate public training evidence')
    return episodes


def validate_rule(rule, item_map):
    if rule.get('kind') not in KINDS or not rule.get('name'):
        raise ValueError('Unknown domain-rule kind or missing name')
    references = rule.get('support_item_ids', [])
    if any(reference not in item_map for reference in references):
        raise ValueError('Rule cites unknown evidence')
    if len({item_map[reference]['episode_id'] for reference in references}) < 2:
        raise ValueError('Rules require at least two independent training episodes')


async def induce_ontology(domain, records, model, excluded_identities=()):
    episodes = public_training_episodes(records,excluded_identities)
    items = {}
    extract_system = 'Extract reusable domain observations using only the provided public training instruction, executed actions, and observations. Return JSON with items containing kind, name, description, and an exact evidence_quote. Do not infer unseen facts.'
    for episode in episodes:
        result = await call_json(model,system_prompt=extract_system,prompt=json.dumps(episode,ensure_ascii=False))
        evidence = json.dumps(episode,ensure_ascii=False)
        for candidate in result.get('items',[]):
            quote = str(candidate.get('evidence_quote',''))
            if candidate.get('kind') not in KINDS or len(quote) < 5 or quote not in evidence:
                continue
            item_id = f"{episode['id'][:16]}_{len(items):04d}"
            items[item_id] = {**candidate,'episode_id':episode['id']}
    if not items:
        raise ValueError('No source-supported training observations')
    aggregate_system = 'Produce domain rules from the supplied items. Each rule must cite at least two item IDs from different episodes. Return JSON with rules containing kind, name, description, support_item_ids, and any required alias/action/steps/condition/insert/before_any_of fields. Observed order alone does not establish a universal prerequisite.'
    result = await call_json(model,system_prompt=aggregate_system,prompt=json.dumps(items,ensure_ascii=False))
    ontology = {'schema_version':1,'domain':domain,'object_roles':{},'state_predicates':{},
                'operation_schemas':{'action_aliases':{}},'transition_schemas':{},'workflow_templates':{},
                'conditional_prerequisites':[],'recovery_policies':[],'rule_provenance':{},
                'evidence_policies':['Domain rules require public evidence from at least two independent training episodes.']}
    for rule in result.get('rules',[]):
        try:
            validate_rule(rule,items)
        except ValueError:
            continue
        kind, name = rule['kind'],rule['name']
        description = str(rule.get('description',''))
        if kind in ('object_role','state_predicate','transition'):
            section = {'object_role':'object_roles','state_predicate':'state_predicates','transition':'transition_schemas'}[kind]
            ontology[section][name] = [description]
        elif kind == 'action_alias':
            if not rule.get('alias') or not rule.get('action'):
                continue
            ontology['operation_schemas']['action_aliases'][rule['alias']] = rule['action']
        elif kind == 'workflow':
            if len(rule.get('steps',[])) < 2:
                continue
            ontology['workflow_templates'][name] = rule['steps']
        elif kind == 'conditional_prerequisite':
            if not rule.get('condition') or not rule.get('insert') or not rule.get('before_any_of'):
                continue
            ontology['conditional_prerequisites'].append({'name':name,'type':'conditional_insertion',
                                                          'condition':rule['condition'],'insert':rule['insert'],
                                                          'before_any_of':rule['before_any_of']})
        else:
            ontology['recovery_policies'].append({'name':name,'policy':description})
        rule_id = kind + ':' + re.sub(r'\W+','_',name.lower())
        ontology['rule_provenance'][rule_id] = {'support_item_ids':rule['support_item_ids'],
                                               'episode_hashes':sorted({items[key]['episode_id'] for key in rule['support_item_ids']}),
                                               'source_kind':'public_train_interaction'}
    if not ontology['rule_provenance']:
        raise ValueError('No evidence-supported domain rules survived')
    ontology['provenance'] = {'source_split':'train','source_episode_count':len(episodes),
                              'frozen_before_online_test':True,'test_time_update':False}
    return ontology
