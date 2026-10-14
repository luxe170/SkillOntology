from __future__ import annotations

import os
from pathlib import Path

from skillontology.benchmark.base import BenchmarkEnvironment, Observation


class Environment(BenchmarkEnvironment):
    def reset_task(self, task):
        if self.backend is None:
            import yaml
            from alfworld.agents.environment import get_environment
            data = self.config.get('data_dir') or os.environ.get('ALFWORLD_DATA_DIR') or os.environ.get('ALFWORLD_DATA')
            base = self.config.get('base_config') or os.environ.get('ALFWORLD_BASE_CONFIG')
            if not data or not base:
                raise ValueError('Set ALFWORLD_DATA_DIR and ALFWORLD_BASE_CONFIG')
            data = Path(data).expanduser().resolve()
            os.environ['ALFWORLD_DATA'] = str(data)
            config = yaml.safe_load(Path(base).read_text(encoding='utf-8'))
            config['general']['random_seed'] = int(self.config.get('seed', 233))
            wrapper = get_environment(config['env']['type'])(config, train_eval=self.config.get('split', 'eval_out_of_distribution'))
            selected = next((path for path in wrapper.game_files if Path(path).resolve().relative_to(data).as_posix() == task.identity['gamefile_id']), None)
            if selected is None:
                raise ValueError('Frozen ALFWorld game identity not found')
            wrapper.game_files = [selected]
            wrapper.num_games = 1
            self.backend = wrapper.init_env(batch_size=1)
        observations, info = self.backend.reset()
        reported = info.get('extra.gamefile', [None])[0]
        if reported and not str(reported).replace('\\', '/').endswith(task.identity['gamefile_id']):
            raise ValueError('ALFWorld task identity mismatch')
        return Observation(str(observations[0]))

    def step_action(self, action):
        observations, rewards, dones, info = self.backend.step([action])
        success = bool(info.get('won', [False])[0])
        return Observation(str(observations[0]), done=bool(dones[0]), success=success, score=100.0 if success else 0.0)
