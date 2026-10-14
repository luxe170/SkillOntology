from __future__ import annotations

from skillontology.benchmark.base import BenchmarkEnvironment, Observation


class Environment(BenchmarkEnvironment):
    def reset_task(self, task):
        if self.backend is None:
            from importlib.metadata import version
            if version('scienceworld') != '1.2.3':
                raise ValueError('ScienceWorld SDK 1.2.3 is required')
            from scienceworld import ScienceWorldEnv
            self.backend = ScienceWorldEnv(envStepLimit=int(self.config.get('max_ticks', 100)))
        names = self.backend.getTaskNames()
        if task.identity['task_name'] not in names:
            raise ValueError('Unknown ScienceWorld task name')
        self.backend.load(str(task.identity['task_name']), int(task.identity['variation_idx']),
                          self.config.get('simplification', 'easy'))
        observation, info = self.backend.reset()
        instruction = info.get('taskDesc')
        if not instruction:
            getter = getattr(self.backend, 'get_task_description', None) or getattr(self.backend, 'getTaskDescription', None)
            if not callable(getter):
                raise ValueError('ScienceWorld did not expose a public task description')
            instruction = getter()
        if str(instruction).strip() != task.instruction.strip():
            raise ValueError('ScienceWorld public instruction mismatch')
        return self._observation(observation, info)

    def _observation(self, text, info, done=False):
        score = float(info.get('score', 0.0))
        return Observation(str(text), done=bool(done), success=score >= 100.0,
                           score=score, ticks=int(info['moves']) if 'moves' in info else None)

    def step_action(self, action):
        observation, _, done, info = self.backend.step(action)
        return self._observation(observation, info, done)
