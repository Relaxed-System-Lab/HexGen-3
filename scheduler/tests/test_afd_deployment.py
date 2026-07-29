import unittest

from simulator.scheduling import (
    AFDDeploymentSpec,
    afd_deployment_spec_to_runtime_launch_spec,
)


class TestAFDDeployment(unittest.TestCase):
    def test_ffn_skips_standalone_server_warmup(self):
        launch = afd_deployment_spec_to_runtime_launch_spec(
            AFDDeploymentSpec(
                model_path="/models/test",
                prefill_gpus=1,
                attention_gpus=1,
                ffn_gpus=1,
            )
        )
        commands = {process.role: process.command for process in launch.processes}

        self.assertIn("--skip-server-warmup", commands["ffn"])
        self.assertNotIn("--skip-server-warmup", commands["prefill"])
        self.assertNotIn("--skip-server-warmup", commands["attention"])


if __name__ == "__main__":
    unittest.main()
