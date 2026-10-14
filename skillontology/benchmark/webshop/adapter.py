from __future__ import annotations
import re
from skillontology.schema import QueryDecomposition, QueryOperationNeed

def parse_query(instruction: str) -> QueryDecomposition:
    public_instruction = ' '.join(str(instruction or '').split())
    price_match = re.search('(?:under|below|less than|lower than|at most|up to)\\s*\\$?\\s*(\\d+(?:\\.\\d+)?)', public_instruction, re.IGNORECASE)
    constraints = ['use only public WebShop observations and controls']
    if price_match:
        constraints.append(f'visible price must not exceed {price_match.group(1)} dollars')
    if re.search('\\b(size|color|colour|flavor|flavour|style|quantity|pack)\\b', public_instruction, re.IGNORECASE):
        constraints.append('select every requested visible buying option before purchase')
    specifications = [('parse', 'public shopping instruction', [], ['product type, public requirements, requested options, and price ceiling identified']), ('search', 'WebShop product catalog', ['public shopping requirements'], ['visible search results containing candidate products']), ('compare', 'visible search-result listings against public instruction constraints', ['visible search results containing candidate products', 'parsed public shopping requirements'], ['best qualifying visible candidate product identified for detail inspection']), ('navigate', 'best qualifying visible candidate from the current search results to its public product page', ['best qualifying visible candidate product identified', 'exact visible product link'], ['public product detail page for the candidate is visible']), ('inspect', 'public product details', ['public product detail page for the candidate is visible'], ['visible title, price, description, features, and options evidence available']), ('verify', 'candidate against public instruction constraints', ['visible product evidence available'], ['candidate satisfies visible task constraints']), ('configure', 'requested visible product options on the public product detail page, not a search-result candidate', ['verified candidate product', 'every explicitly requested visible size, color, style, quantity, pack, or variant value'], ['product is configured with every explicitly requested visible option clicked and selected before purchase']), ('purchase', 'verified candidate product', ['candidate satisfies visible task constraints', 'requested options selected'], ['purchase submitted through the public Buy Now control'])]
    operations = [QueryOperationNeed(name=f'{action} {obj}', action=action, object=obj, inputs=inputs, outputs=outputs, tools=['WebShop public text interface'], constraints=constraints, priority=index) for index, (action, obj, inputs, outputs) in enumerate(specifications, start=1)]
    return QueryDecomposition(raw_query=public_instruction, normalized_goal=public_instruction, task_type='webshop-public-product-selection', domain_objects=['WebShop', 'public product page', public_instruction], input_artifacts=['public shopping instruction'], expected_outputs=['best visible matching product purchased through Buy Now'], required_operations=operations, tool_hints=['search[...]', 'click[...]'], constraints=constraints, validation_signals=['public page changes after an action', 'Buy Now completes the episode'])
from skillontology.benchmark.base import BenchmarkAdapter

class Adapter(BenchmarkAdapter):
    name = 'webshop'

    def parse_task(self, task, observation, ontology):
        return parse_query(task.instruction)

    def binding_roles(self, task, query):
        return {'goal': task.instruction}
