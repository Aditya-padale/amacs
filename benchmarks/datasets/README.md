# Benchmark datasets

`fixture.json` is the deterministic offline regression fixture used by CI. It is
not a quality dataset. Live benchmark runs should use pinned, locally downloaded
copies of documented public task sets (for example [MMLU](https://huggingface.co/datasets/cais/mmlu),
[HotpotQA](https://hotpotqa.github.io/), and [HumanEval](https://github.com/openai/human-eval)), with their
license, revision, split, preprocessing script, model identifiers, and random
seeds recorded beside the results. The harness intentionally does not download
or silently redistribute those datasets.
