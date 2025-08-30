## SHALE: A Scalable Benchmark for Fine-grained Hallucination Evaluation in LVLMs

**SHALE**  is a **S**calable **HAL**lucination **E**valuation benchmark built on an automated data construction pipeline. It is designed to assess both faithfulness and factuality hallucinations via a fine-grained hallucination categorization scheme. SHALE comprises over 30K image-instruction pairs spanning 12 representative visual perception aspects for faithfulness and 6 knowledge domains for factuality, considering both clean and noisy scenarios.


### Evaluation Data

The evaluation data can be downloaded from this [LINK](https://1drv.ms/u/c/3990e975c588b26f/EWkj2U7q0fdDuAhQHjpVwt4BlNbFa1T1pz6OZy3usscbtw).

The evaluation data is structured in JSON format, as follows:

```python
[
  {
    "id": 1,
    "task": "PosYNQ",
    "hallucination_type": "faithfulness",
    "evaluation_dimension": "type",
    "entity_type": "object",
    "images": [
      "./Image/1.jpg"
    ],
    "instruction": "Is the object in the image a pan?",
    "ground_truth": "Yes"
  },
    ...
]
```

The images are saved in `./Image`, and the instructions are saved in `./Query`.

### Evaluation Code

The code for model evaluation is saved in `./Evaluation`.

### Construction Code

The code for data construction is currently being organized and will be released very soon!
