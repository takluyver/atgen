# ATGen: Active Text Generation

## How to launch without config

```bash
python3 run_active_learning.py +experiment.num_labeled=8 +experiment.num_al_iterations=2 +experiment.al_query_size=2 +data.dataset=SpeedOfMagic/gigaword_tiny +data.input_column_name=document +data.output_column_name=summary +model.name=TinyLlama/TinyLlama-iter_1.1B-Chat-v1.0 +labeler.type=golden +experiment.al_strategy=random
```