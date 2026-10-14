from __future__ import annotations

import os
from pathlib import Path
import random
import sys

from skillontology.benchmark.base import BenchmarkEnvironment, Observation


class Environment(BenchmarkEnvironment):
    def reset_task(self, task):
        if self.backend is None:
            if os.environ.get('PYTHONHASHSEED') != '0':
                raise ValueError('Start Python with PYTHONHASHSEED=0 for the frozen WebShop catalog')
            root = self.config.get('repo') or os.environ.get('WEBSHOP_REPO')
            if not root or not (Path(root) / 'web_agent_site').is_dir():
                raise ValueError('Set WEBSHOP_REPO to the official checkout')
            if str(Path(root).resolve()) not in sys.path:
                sys.path.insert(0, str(Path(root).resolve()))
            from web_agent_site.envs.web_agent_text_env import SimServer, WebAgentTextEnv
            from web_agent_site.utils import DEFAULT_FILE_PATH
            random.seed(int(self.config.get('seed', 233)))
            server = SimServer('http://127.0.0.1:3000', self.config.get('products') or DEFAULT_FILE_PATH,
                               num_products=self.config.get('num_products'), human_goals=True, show_attrs=False)
            self.backend = WebAgentTextEnv(observation_mode=self.config.get('observation_mode', 'text_rich'), server=server, show_attrs=False)
        observation, _ = self.backend.reset(session=int(task.identity['task_id']))
        if self.backend.instruction_text != task.instruction:
            raise ValueError('WebShop public instruction mismatch')
        return self._observation(observation)

    def _observation(self, text, done=False, reward=0.0):
        actions = self.backend.get_available_actions()
        controls = tuple(map(str, actions.get('clickables', [])))
        success = float(reward) >= 1.0 - 1e-9
        return Observation(str(text), controls, bool(done), success, float(reward) * 100.0)

    def step_action(self, action):
        observation, reward, done, _ = self.backend.step(action)
        return self._observation(observation, done, reward)
