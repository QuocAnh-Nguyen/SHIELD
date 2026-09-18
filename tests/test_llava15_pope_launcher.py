import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY_ROOT / "experiments/scripts/run_llava15_pope_coco.sh"


class LLaVA15PopeLauncherTests(unittest.TestCase):
    def test_dry_run_uses_configured_paths_and_author_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "server.env"
            config_path.write_text(
                "\n".join(
                    [
                        "MODEL_PATH=/models/llava-v1.5-7b",
                        "COCO_IMAGE_DIR=/datasets/coco/val2014",
                        "POPE_DATA_DIR=/datasets/pope/coco",
                        "POPE_CAPTION_FILE=/datasets/captions/pope.jsonl",
                        "OUTPUT_DIR=/results/shield",
                        "CUDA_VISIBLE_DEVICES=3",
                        "SEED=42",
                        "CD_ALPHA=2.0",
                        "CD_BETA=0.35",
                        "NOISE_STEP=999",
                        "THE=0.011",
                        "GAMMA_GAIN=3.0",
                        "GAMMA_REDUCE=3.0",
                        "GAIN_PER=0.5",
                        "REDUCE_PER=0.0",
                        "BIAS_WEIGHT=0.1",
                        "BIAS_SAMPLE_NUM=32",
                        "CW_EPSILON=0.14",
                        "CW_NUM_STEPS=30",
                        "CW_C=12",
                        "CW_LR=0.14",
                        "MAX_NEW_TOKENS=1024",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    "bash",
                    str(LAUNCHER),
                    "--config",
                    str(config_path),
                    "--split",
                    "adversarial",
                    "--dry-run",
                ],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("CUDA_VISIBLE_DEVICES=3", completed.stdout)
        self.assertIn("--model-path /models/llava-v1.5-7b", completed.stdout)
        self.assertIn("--question-file /datasets/pope/coco/coco_pope_adversarial.json", completed.stdout)
        self.assertIn("--image-folder /datasets/coco/val2014", completed.stdout)
        self.assertIn("--caption-file /datasets/captions/pope.jsonl", completed.stdout)
        self.assertIn("--cd_alpha 2.0", completed.stdout)
        self.assertIn("--bias_sample_num 32", completed.stdout)
        self.assertIn("--max-new-tokens 1024", completed.stdout)
        self.assertIn("/results/shield/llava15_coco_pope_adversarial", completed.stdout)


class LLaVA15ChairLauncherTests(unittest.TestCase):
    def test_dry_run_exports_hf_home_when_configured(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "server.env"
            config_path.write_text(
                "\n".join(
                    [
                        "MODEL_PATH=/models/llava-v1.5-7b",
                        "COCO_IMAGE_DIR=/datasets/coco/val2014",
                        "POPE_DATA_DIR=/datasets/pope/coco",
                        "POPE_CAPTION_FILE=/datasets/captions/pope.jsonl",
                        "OUTPUT_DIR=/results/shield",
                        "CUDA_VISIBLE_DEVICES=0",
                        "SEED=42",
                        "CD_ALPHA=2.0",
                        "CD_BETA=0.35",
                        "NOISE_STEP=999",
                        "THE=0.011",
                        "GAMMA_GAIN=3.0",
                        "GAMMA_REDUCE=3.0",
                        "GAIN_PER=0.5",
                        "REDUCE_PER=0.0",
                        "BIAS_WEIGHT=0.1",
                        "BIAS_SAMPLE_NUM=32",
                        "CW_EPSILON=0.14",
                        "CW_NUM_STEPS=30",
                        "CW_C=12",
                        "CW_LR=0.14",
                        "MAX_NEW_TOKENS=1024",
                        "HF_HOME=/data/hf_cache",
                        "HF_HUB_DISABLE_XET=1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    "bash",
                    str(LAUNCHER),
                    "--config",
                    str(config_path),
                    "--dry-run",
                ],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("HF_HOME=/data/hf_cache", completed.stdout)
        self.assertIn("HF_HUB_DISABLE_XET=1", completed.stdout)

    def test_dry_run_uses_chair_protocol_and_author_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "chair.env"
            config_path.write_text(
                "\n".join(
                    [
                        "MODEL_PATH=/models/llava-v1.5-7b",
                        "COCO_IMAGE_DIR=/datasets/coco/val2014",
                        "CHAIR_QUESTION_FILE=/datasets/chair/questions.jsonl",
                        "CHAIR_CAPTION_FILE=/datasets/captions/chair.jsonl",
                        "CHAIR_CACHE_FILE=/datasets/chair/chair.pkl",
                        "OUTPUT_DIR=/results/shield",
                        "CUDA_VISIBLE_DEVICES=3",
                        "SEED=42",
                        "CD_ALPHA=2.0",
                        "CD_BETA=0.35",
                        "NOISE_STEP=500",
                        "THE=0.002",
                        "GAMMA_GAIN=3.0",
                        "GAMMA_REDUCE=3.0",
                        "GAIN_PER=0.55",
                        "REDUCE_PER=0.0",
                        "BIAS_WEIGHT=0.01",
                        "BIAS_SAMPLE_NUM=32",
                        "CW_EPSILON=0.14",
                        "CW_NUM_STEPS=30",
                        "CW_C=12",
                        "CW_LR=0.02",
                        "MAX_NEW_TOKENS=128",
                        'PROMPT="Describe this image."',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    "bash",
                    str(REPOSITORY_ROOT / "experiments/scripts/run_llava15_chair.sh"),
                    "--config",
                    str(config_path),
                    "--dry-run",
                ],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--question-file /datasets/chair/questions.jsonl", completed.stdout)
        self.assertIn("--image-folder /datasets/coco/val2014", completed.stdout)
        self.assertIn("--caption-file /datasets/captions/chair.jsonl", completed.stdout)
        self.assertIn("--prompt Describe\\ this\\ image.", completed.stdout)
        self.assertIn("--max-new-tokens 128", completed.stdout)
        self.assertIn("--cw_lr 0.02", completed.stdout)
        self.assertIn("--cache /datasets/chair/chair.pkl", completed.stdout)


class LLaVA15BeafLauncherTests(unittest.TestCase):
    def test_dry_run_prints_caption_inference_and_metric_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "beaf.env"
            config_path.write_text(
                "\n".join(
                    [
                        "MODEL_PATH=/models/llava-v1.5-7b",
                        "BEAF_IMAGE_DIR=/datasets/beaf",
                        "BEAF_QNA_FILE=/datasets/beaf/beaf_qna.json",
                        "BEAF_CAPTION_FILE=/results/captions/beaf.jsonl",
                        "OUTPUT_DIR=/results/shield",
                        "CUDA_VISIBLE_DEVICES=0",
                        "SEED=42",
                        "CD_ALPHA=2.0",
                        "CD_BETA=0.35",
                        "NOISE_STEP=999",
                        "THE=0.011",
                        "GAMMA_GAIN=3.0",
                        "GAMMA_REDUCE=3.0",
                        "GAIN_PER=0.5",
                        "REDUCE_PER=0.0",
                        "BIAS_WEIGHT=0.1",
                        "BIAS_SAMPLE_NUM=32",
                        "CW_EPSILON=0.14",
                        "CW_NUM_STEPS=30",
                        "CW_C=12",
                        "CW_LR=0.14",
                        "MAX_NEW_TOKENS=1024",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    "bash",
                    str(REPOSITORY_ROOT / "experiments/scripts/run_llava15_beaf.sh"),
                    "--config",
                    str(config_path),
                    "--dry-run",
                ],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("generate_first_captions_llava.py", completed.stdout)
        self.assertIn("--question-file /datasets/beaf/beaf_qna.json", completed.stdout)
        self.assertIn("--output-file /results/captions/beaf.jsonl", completed.stdout)
        self.assertIn("--max-new-tokens 70", completed.stdout)
        self.assertNotIn("--orig-only", completed.stdout)
        self.assertIn("--image-folder /datasets/beaf", completed.stdout)
        self.assertIn("--caption-file /results/captions/beaf.jsonl", completed.stdout)
        self.assertIn("--cd_alpha 2.0", completed.stdout)
        self.assertIn("--max-new-tokens 1024", completed.stdout)
        self.assertIn("beaf_metric.py", completed.stdout)
        self.assertIn("--model-answers /results/shield/llava15_beaf_answers_seed42.json", completed.stdout)


class LLaVA15CausalHalLauncherTests(unittest.TestCase):
    def test_dry_run_prints_caption_inference_and_metric_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "causalhal.env"
            config_path.write_text(
                "\n".join(
                    [
                        "MODEL_PATH=/models/llava-v1.5-7b",
                        "CAUSALHAL_IMAGE_DIR=/datasets/causalhal/images",
                        "CAUSALHAL_QA_FILE=/datasets/causalhal/qa.json",
                        "CAUSALHAL_CAPTION_FILE=/results/captions/causalhal.jsonl",
                        "OUTPUT_DIR=/results/shield",
                        "CUDA_VISIBLE_DEVICES=0",
                        "SEED=42",
                        "CD_ALPHA=2.0",
                        "CD_BETA=0.35",
                        "NOISE_STEP=999",
                        "THE=0.011",
                        "GAMMA_GAIN=3.0",
                        "GAMMA_REDUCE=3.0",
                        "GAIN_PER=0.5",
                        "REDUCE_PER=0.0",
                        "BIAS_WEIGHT=0.1",
                        "BIAS_SAMPLE_NUM=32",
                        "CW_EPSILON=0.14",
                        "CW_NUM_STEPS=30",
                        "CW_C=12",
                        "CW_LR=0.14",
                        "MAX_NEW_TOKENS=1024",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    "bash",
                    str(REPOSITORY_ROOT / "experiments/scripts/run_llava15_causalhal.sh"),
                    "--config",
                    str(config_path),
                    "--dry-run",
                ],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("generate_first_captions_llava.py", completed.stdout)
        self.assertIn("--question-file /datasets/causalhal/qa.json", completed.stdout)
        self.assertIn("--output-file /results/captions/causalhal.jsonl", completed.stdout)
        self.assertIn("--prompt Describe\\ this\\ image.", completed.stdout)
        self.assertIn("--max-new-tokens 128", completed.stdout)
        self.assertIn("--image-folder /datasets/causalhal/images", completed.stdout)
        self.assertIn("--caption-file /results/captions/causalhal.jsonl", completed.stdout)
        self.assertIn("--cd_alpha 2.0", completed.stdout)
        self.assertIn("--max-new-tokens 1024", completed.stdout)
        self.assertIn("causalhal_metric.py", completed.stdout)
        self.assertIn("--qa-file /datasets/causalhal/qa.json", completed.stdout)
        self.assertIn("--resp-file /results/shield/llava15_causalhal_answers_seed42.json", completed.stdout)


class Qwen2VLCausalHalLauncherTests(unittest.TestCase):
    def test_dry_run_prints_caption_inference_and_metric_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "qwen2vl_causalhal.env"
            config_path.write_text(
                "\n".join(
                    [
                        "MODEL_PATH=Qwen/Qwen2-VL-7B-Instruct",
                        "CAUSALHAL_IMAGE_DIR=/datasets/causalhal/images",
                        "CAUSALHAL_QA_FILE=/datasets/causalhal/qa.json",
                        "CAUSALHAL_CAPTION_FILE=/results/captions/qwen2vl_causalhal.jsonl",
                        "OUTPUT_DIR=/results/shield",
                        "CUDA_VISIBLE_DEVICES=0",
                        "SEED=42",
                        "CD_ALPHA=2.0",
                        "CD_BETA=0.35",
                        "NOISE_STEP=999",
                        "THE=0.011",
                        "GAMMA_GAIN=3.0",
                        "GAMMA_REDUCE=3.0",
                        "GAIN_PER=0.5",
                        "REDUCE_PER=0.0",
                        "BIAS_WEIGHT=0.1",
                        "BIAS_SAMPLE_NUM=32",
                        "CW_EPSILON=0.14",
                        "CW_NUM_STEPS=30",
                        "CW_C=12",
                        "CW_LR=0.14",
                        "MAX_NEW_TOKENS=1024",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    "bash",
                    str(REPOSITORY_ROOT / "experiments/scripts/run_qwen2vl_causalhal.sh"),
                    "--config",
                    str(config_path),
                    "--dry-run",
                ],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("generate_first_captions_qwen2vl.py", completed.stdout)
        self.assertIn("--question-file /datasets/causalhal/qa.json", completed.stdout)
        self.assertIn("--output-file /results/captions/qwen2vl_causalhal.jsonl", completed.stdout)
        self.assertIn("--prompt Describe\\ this\\ image.", completed.stdout)
        self.assertIn("--max-new-tokens 128", completed.stdout)
        self.assertIn("--image-folder /datasets/causalhal/images", completed.stdout)
        self.assertIn("--caption-file /results/captions/qwen2vl_causalhal.jsonl", completed.stdout)
        self.assertIn("--cd_alpha 2.0", completed.stdout)
        self.assertIn("--max-new-tokens 1024", completed.stdout)
        self.assertIn("causalhal_metric.py", completed.stdout)
        self.assertIn("--qa-file /datasets/causalhal/qa.json", completed.stdout)
        self.assertIn("--resp-file /results/shield/qwen2vl_causalhal_answers_seed42.json", completed.stdout)


if __name__ == "__main__":
    unittest.main()
