from __future__ import annotations

import unittest

from eureka_lite.mjwarp_ant import MjwarpAntConfig, run_mjwarp_ant


class MjwarpAntTests(unittest.TestCase):
    def test_rejects_invalid_world_count_before_importing_gpu_dependencies(self) -> None:
        with self.assertRaisesRegex(ValueError, "worlds"):
            run_mjwarp_ant(MjwarpAntConfig(worlds=0))

    def test_rejects_random_each_step_with_cuda_graph(self) -> None:
        config = MjwarpAntConfig(action_mode="random-each-step", use_cuda_graph=True)
        with self.assertRaisesRegex(ValueError, "CUDA graph"):
            run_mjwarp_ant(config)


if __name__ == "__main__":
    unittest.main()
