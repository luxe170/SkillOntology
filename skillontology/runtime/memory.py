class InteractionMemory:
    def __init__(self, system_prompt, task, guidance, observation):
        self.system_prompt = system_prompt
        self.task = task
        self.guidance = guidance
        self.initial = observation
        self.turns = []

    def _public_message(self, observation):
        text = 'Task:\n' + self.task + '\n\nGuidance:\n' + self.guidance + '\n\nObservation:\n' + observation.text
        if observation.controls:
            text += '\n\nPublic controls:\n' + '\n'.join(observation.controls)
        return text

    def messages(self, history_turns):
        turns = self.turns if history_turns is None else self.turns[-history_turns:]
        messages = [{'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': self._public_message(self.initial)}]
        for response, observation, feedback in turns:
            messages.append({'role': 'assistant', 'content': response})
            messages.append({'role': 'user', 'content': self._public_message(observation) + ('\n\n' + feedback if feedback else '')})
        return messages

    def record(self, response, observation, feedback=''):
        self.turns.append((response, observation, feedback))
