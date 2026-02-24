# codechat-eval

This repo evaluates the quality of LLM-generated code in real-world developer conversations, using the [CodeChat-V2.0](https://huggingface.co/datasets/Suzhen/CodeChat-V2.0) dataset (587k conversations from [WildChat](https://huggingface.co/datasets/allenai/WildChat)).

Here, we filter and augment the database with syntactic & semantic analysis findings.

### Scripts


> [!IMPORTANT]
> Requires Python 3.13+


| Script                | Description                                |
| --------------------- | ------------------------------------------ |
| `scripts/download.py` | Download the CodeChat-V2.0 dataset         |
| `scripts/filter.py`   | Filter to English prompts with Python code |
| `scripts/syntax.py`   | Syntactic analysis (ruff & radon)          |
| `scripts/semantic.py` | Semantic analysis of prompt-code pairs     |

### Citation

```
@misc{zhong2025developerllmconversationsempiricalstudy,
      title={Developer-LLM Conversations: An Empirical Study of Interactions and Generated Code Quality},
      author={Suzhen Zhong and Ying Zou and Bram Adams},
      year={2025},
      eprint={2509.10402},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2509.10402},
}
```

### License

[MIT](https://github.com/darragh0/codechat-eval?tab=MIT-1-ov-file)
