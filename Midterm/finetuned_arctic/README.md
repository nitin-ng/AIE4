---
base_model: Snowflake/snowflake-arctic-embed-m
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
- dot_accuracy@1
- dot_accuracy@3
- dot_accuracy@5
- dot_accuracy@10
- dot_precision@1
- dot_precision@3
- dot_precision@5
- dot_precision@10
- dot_recall@1
- dot_recall@3
- dot_recall@5
- dot_recall@10
- dot_ndcg@10
- dot_mrr@10
- dot_map@100
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:600
- loss:MatryoshkaLoss
- loss:MultipleNegativesRankingLoss
widget:
- source_sentence: How should explanations of AI system impacts be tailored to different
    users and levels of risk?
  sentences:
  - "login is not available, identity verification may need to be performed before\
    \ providing such a report to ensure \nuser privacy. Additionally, summary reporting\
    \ should be proactively made public with general information \nabout how peoples’\
    \ data and metadata is used, accessed, and stored. Summary reporting should include\
    \ the"
  - "NOTICE & \nEXPLANATION \nWHY THIS PRINCIPLE IS IMPORTANT\nThis section provides\
    \ a brief summary of the problems which the principle seeks to address and protect\
    \ \nagainst, including illustrative examples. \n•\nA predictive policing system\
    \ claimed to identify individuals at greatest risk to commit or become the victim\
    \ of\ngun violence (based on automated analysis of social ties to gang members,\
    \ criminal histories, previous experi­\nences of gun violence, and other factors)\
    \ and led to individuals being placed on a watch list with no\nexplanation or\
    \ public transparency regarding how the system came to its conclusions.85 Both\
    \ police and\nthe public deserve to understand why and how such a system is making\
    \ these determinations.\n•"
  - "ations, including: the responsible entities for accountability purposes; the\
    \ goal and use cases for the system, \nidentified users, and impacted populations;\
    \ the assessment of notice clarity and timeliness; the assessment of \nthe explanation's\
    \ validity and accessibility; the assessment of the level of risk; and the account\
    \ and assessment \nof how explanations are tailored, including to the purpose,\
    \ the recipient of the explanation, and the level of \nrisk. Individualized profile\
    \ information should be made readily available to the greatest extent possible\
    \ that \nincludes explanations for any system impacts or inferences. Reporting\
    \ should be provided in a clear plain \nlanguage and machine-readable manner.\
    \ \n44"
- source_sentence: What are some practical approaches that can be implemented to ensure
    compliance with laws like the Biometric Information Privacy Act?
  sentences:
  - "NOTICE & \nEXPLANATION \nHOW THESE PRINCIPLES CAN MOVE INTO PRACTICE\nReal-life\
    \ examples of how these principles can become reality, through laws, policies,\
    \ and practical \ntechnical and sociotechnical approaches to protecting rights,\
    \ opportunities, and access. ­­­­­\nPeople in Illinois are given written notice\
    \ by the private sector if their biometric informa-\ntion is used. The Biometric\
    \ Information Privacy Act enacted by the state contains a number of provisions\
    \ \nconcerning the use of individual biometric data and identifiers. Included\
    \ among them is a provision that no private \nentity may \"collect, capture, purchase,\
    \ receive through trade, or otherwise obtain\" such information about an"
  - "NOTICE & \nEXPLANATION \nHOW THESE PRINCIPLES CAN MOVE INTO PRACTICE\nReal-life\
    \ examples of how these principles can become reality, through laws, policies,\
    \ and practical \ntechnical and sociotechnical approaches to protecting rights,\
    \ opportunities, and access. ­­­­­\nPeople in Illinois are given written notice\
    \ by the private sector if their biometric informa-\ntion is used. The Biometric\
    \ Information Privacy Act enacted by the state contains a number of provisions\
    \ \nconcerning the use of individual biometric data and identifiers. Included\
    \ among them is a provision that no private \nentity may \"collect, capture, purchase,\
    \ receive through trade, or otherwise obtain\" such information about an"
  - 'the community, both those living in the housing complex and not, to have videos
    of them sent to the local

    police department and made available for scanning by its facial recognition software.66

    •'
- source_sentence: 'How does the collection of personal data from social media by
    insurers raise ethical concerns regarding data privacy?  '
  sentences:
  - "assessment and mitigation. Organizational stakeholders including those with oversight\
    \ of the business process \nor operation being automated, as well as other organizational\
    \ divisions that may be affected due to the use of \nthe system, should be involved\
    \ in establishing governance procedures. Responsibility should rest high enough\
    \ \nin the organization that decisions about resources, mitigation, incident response,\
    \ and potential rollback can be \nmade promptly, with sufficient weight given\
    \ to risk mitigation objectives against competing concerns. Those \nholding this\
    \ responsibility should be made aware of any use cases with the potential for\
    \ meaningful impact on"
  - "systems should be evaluated, protected against, and redressed at both the individual\
    \ and community levels. \nEQUITY: “Equity” means the consistent and systematic\
    \ fair, just, and impartial treatment of all individuals. \nSystemic, fair, and\
    \ just treatment must take into account the status of individuals who belong to\
    \ underserved \ncommunities that have been denied such treatment, such as Black,\
    \ Latino, and Indigenous and Native American \npersons, Asian Americans and Pacific\
    \ Islanders and other persons of color; members of religious minorities; \nwomen,\
    \ girls, and non-binary people; lesbian, gay, bisexual, transgender, queer, and\
    \ intersex (LGBTQI+)"
  - "DATA PRIVACY \nWHY THIS PRINCIPLE IS IMPORTANT\nThis section provides a brief\
    \ summary of the problems which the principle seeks to address and protect \n\
    against, including illustrative examples. \n•\nAn insurer might collect data from\
    \ a person's social media presence as part of deciding what life\ninsurance rates\
    \ they should be offered.64\n•\nA data broker harvested large amounts of personal\
    \ data and then suffered a breach, exposing hundreds of\nthousands of people to\
    \ potential identity theft. 65\n•\nA local public housing authority installed\
    \ a facial recognition system at the entrance to housing complexes to\nassist\
    \ law enforcement with identifying individuals viewed via camera when police reports\
    \ are filed, leading"
- source_sentence: 'What is the purpose of the technical companion in relation to
    automated systems?  '
  sentences:
  - "NOTICE & \nEXPLANATION \nWHAT SHOULD BE EXPECTED OF AUTOMATED SYSTEMS\nThe expectations\
    \ for automated systems are meant to serve as a blueprint for the development\
    \ of additional \ntechnical standards and practices that are tailored for particular\
    \ sectors and contexts. \nAn automated system should provide demonstrably clear,\
    \ timely, understandable, and accessible notice of use, and \nexplanations as\
    \ to how and why a decision was made or an action was taken by the system. These\
    \ expectations are \nexplained below. \nProvide clear, timely, understandable,\
    \ and accessible notice of use and explanations ­\nGenerally accessible plain\
    \ language documentation. The entity responsible for using the automated"
  - "tion or implemented under existing U.S. laws. For example, government surveillance,\
    \ and data search and \nseizure are subject to legal requirements and judicial\
    \ oversight. There are Constitutional requirements for \nhuman review of criminal\
    \ investigative matters and statutory requirements for judicial review. Civil\
    \ rights laws \nprotect the American people against discrimination. \n8"
  - "technical companion is intended to be used as a reference by people across many\
    \ circumstances – anyone \nimpacted by automated systems, and anyone developing,\
    \ designing, deploying, evaluating, or making policy to \ngovern the use of an\
    \ automated system. \nEach principle is accompanied by three supplemental sections:\
    \ \n1\n2\nWHY THIS PRINCIPLE IS IMPORTANT: \nThis section provides a brief summary\
    \ of the problems that the principle seeks to address and protect against, including\
    \ \nillustrative examples. \nWHAT SHOULD BE EXPECTED OF AUTOMATED SYSTEMS: \n\
    • The expectations for automated systems are meant to serve as a blueprint for\
    \ the development of additional technical\nstandards and practices that should\
    \ be tailored for particular sectors and contexts."
- source_sentence: How does the requirement for human alternatives in automated systems
    contribute to protecting the public from harmful impacts?
  sentences:
  - "About this Document \nThe Blueprint for an AI Bill of Rights: Making Automated\
    \ Systems Work for the American People was \npublished by the White House Office\
    \ of Science and Technology Policy in October 2022. This framework was \nreleased\
    \ one year after OSTP announced the launch of a process to develop “a bill of\
    \ rights for an AI-powered \nworld.” Its release follows a year of public engagement\
    \ to inform this initiative. The framework is available \nonline at: https://www.whitehouse.gov/ostp/ai-bill-of-rights\
    \ \nAbout the Office of Science and Technology Policy \nThe Office of Science\
    \ and Technology Policy (OSTP) was established by the National Science and Technology"
  - "automated systems make on underserved communities and to institute proactive\
    \ protections that support these \ncommunities. \n•\nAn automated system using\
    \ nontraditional factors such as educational attainment and employment history\
    \ as"
  - "SECTION TITLE\nHUMAN ALTERNATIVES, CONSIDERATION, AND FALLBACK\nYou should be\
    \ able to opt out, where appropriate, and have access to a person who can quickly\
    \ \nconsider and remedy problems you encounter. You should be able to opt out\
    \ from automated systems in \nfavor of a human alternative, where appropriate.\
    \ Appropriateness should be determined based on reasonable \nexpectations in a\
    \ given context and with a focus on ensuring broad accessibility and protecting\
    \ the public from \nespecially harmful impacts. In some cases, a human or other\
    \ alternative may be required by law. You should have \naccess to timely human\
    \ consideration and remedy by a fallback and escalation process if an automated\
    \ system"
model-index:
- name: SentenceTransformer based on Snowflake/snowflake-arctic-embed-m
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: Unknown
      type: unknown
    metrics:
    - type: cosine_accuracy@1
      value: 0.71
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.89
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.93
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.97
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.71
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.29666666666666663
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.18599999999999994
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.09699999999999999
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.71
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.89
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.93
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.97
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.8448032235507587
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.8038452380952382
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.8057425642984465
      name: Cosine Map@100
    - type: dot_accuracy@1
      value: 0.71
      name: Dot Accuracy@1
    - type: dot_accuracy@3
      value: 0.89
      name: Dot Accuracy@3
    - type: dot_accuracy@5
      value: 0.93
      name: Dot Accuracy@5
    - type: dot_accuracy@10
      value: 0.97
      name: Dot Accuracy@10
    - type: dot_precision@1
      value: 0.71
      name: Dot Precision@1
    - type: dot_precision@3
      value: 0.29666666666666663
      name: Dot Precision@3
    - type: dot_precision@5
      value: 0.18599999999999994
      name: Dot Precision@5
    - type: dot_precision@10
      value: 0.09699999999999999
      name: Dot Precision@10
    - type: dot_recall@1
      value: 0.71
      name: Dot Recall@1
    - type: dot_recall@3
      value: 0.89
      name: Dot Recall@3
    - type: dot_recall@5
      value: 0.93
      name: Dot Recall@5
    - type: dot_recall@10
      value: 0.97
      name: Dot Recall@10
    - type: dot_ndcg@10
      value: 0.8448032235507587
      name: Dot Ndcg@10
    - type: dot_mrr@10
      value: 0.8038452380952382
      name: Dot Mrr@10
    - type: dot_map@100
      value: 0.8057425642984465
      name: Dot Map@100
---

# SentenceTransformer based on Snowflake/snowflake-arctic-embed-m

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [Snowflake/snowflake-arctic-embed-m](https://huggingface.co/Snowflake/snowflake-arctic-embed-m). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [Snowflake/snowflake-arctic-embed-m](https://huggingface.co/Snowflake/snowflake-arctic-embed-m) <!-- at revision e2b128b9fa60c82b4585512b33e1544224ffff42 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 tokens
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'How does the requirement for human alternatives in automated systems contribute to protecting the public from harmful impacts?',
    'SECTION TITLE\nHUMAN ALTERNATIVES, CONSIDERATION, AND FALLBACK\nYou should be able to opt out, where appropriate, and have access to a person who can quickly \nconsider and remedy problems you encounter. You should be able to opt out from automated systems in \nfavor of a human alternative, where appropriate. Appropriateness should be determined based on reasonable \nexpectations in a given context and with a focus on ensuring broad accessibility and protecting the public from \nespecially harmful impacts. In some cases, a human or other alternative may be required by law. You should have \naccess to timely human consideration and remedy by a fallback and escalation process if an automated system',
    'automated systems make on underserved communities and to institute proactive protections that support these \ncommunities. \n•\nAn automated system using nontraditional factors such as educational attainment and employment history as',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval

* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.71       |
| cosine_accuracy@3   | 0.89       |
| cosine_accuracy@5   | 0.93       |
| cosine_accuracy@10  | 0.97       |
| cosine_precision@1  | 0.71       |
| cosine_precision@3  | 0.2967     |
| cosine_precision@5  | 0.186      |
| cosine_precision@10 | 0.097      |
| cosine_recall@1     | 0.71       |
| cosine_recall@3     | 0.89       |
| cosine_recall@5     | 0.93       |
| cosine_recall@10    | 0.97       |
| cosine_ndcg@10      | 0.8448     |
| cosine_mrr@10       | 0.8038     |
| **cosine_map@100**  | **0.8057** |
| dot_accuracy@1      | 0.71       |
| dot_accuracy@3      | 0.89       |
| dot_accuracy@5      | 0.93       |
| dot_accuracy@10     | 0.97       |
| dot_precision@1     | 0.71       |
| dot_precision@3     | 0.2967     |
| dot_precision@5     | 0.186      |
| dot_precision@10    | 0.097      |
| dot_recall@1        | 0.71       |
| dot_recall@3        | 0.89       |
| dot_recall@5        | 0.93       |
| dot_recall@10       | 0.97       |
| dot_ndcg@10         | 0.8448     |
| dot_mrr@10          | 0.8038     |
| dot_map@100         | 0.8057     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 600 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 600 samples:
  |         | sentence_0                                                                        | sentence_1                                                                         |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             |
  | details | <ul><li>min: 9 tokens</li><li>mean: 20.13 tokens</li><li>max: 31 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 84.19 tokens</li><li>max: 160 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                         | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>What are the key principles outlined in the AI Bill of Rights aimed at ensuring automated systems benefit the American people?  </code>      | <code>BLUEPRINT FOR AN <br>AI BILL OF <br>RIGHTS <br>MAKING AUTOMATED <br>SYSTEMS WORK FOR <br>THE AMERICAN PEOPLE <br>OCTOBER 2022</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  | <code>How does the AI Bill of Rights address potential ethical concerns related to automated systems?</code>                                       | <code>BLUEPRINT FOR AN <br>AI BILL OF <br>RIGHTS <br>MAKING AUTOMATED <br>SYSTEMS WORK FOR <br>THE AMERICAN PEOPLE <br>OCTOBER 2022</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  | <code>What is the purpose of the Blueprint for an AI Bill of Rights published by the White House Office of Science and Technology Policy?  </code> | <code>About this Document <br>The Blueprint for an AI Bill of Rights: Making Automated Systems Work for the American People was <br>published by the White House Office of Science and Technology Policy in October 2022. This framework was <br>released one year after OSTP announced the launch of a process to develop “a bill of rights for an AI-powered <br>world.” Its release follows a year of public engagement to inform this initiative. The framework is available <br>online at: https://www.whitehouse.gov/ostp/ai-bill-of-rights <br>About the Office of Science and Technology Policy <br>The Office of Science and Technology Policy (OSTP) was established by the National Science and Technology</code> |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          768,
          512,
          256,
          128,
          64
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 20
- `per_device_eval_batch_size`: 20
- `num_train_epochs`: 5
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 20
- `per_device_eval_batch_size`: 20
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 5
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `eval_use_gather_object`: False
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | cosine_map@100 |
|:------:|:----:|:--------------:|
| 1.0    | 30   | 0.7871         |
| 1.6667 | 50   | 0.7966         |
| 2.0    | 60   | 0.7986         |
| 3.0    | 90   | 0.8057         |


### Framework Versions
- Python: 3.11.9
- Sentence Transformers: 3.2.0
- Transformers: 4.44.2
- PyTorch: 2.4.1
- Accelerate: 0.34.2
- Datasets: 3.0.1
- Tokenizers: 0.19.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MatryoshkaLoss
```bibtex
@misc{kusupati2024matryoshka,
    title={Matryoshka Representation Learning},
    author={Aditya Kusupati and Gantavya Bhatt and Aniket Rege and Matthew Wallingford and Aditya Sinha and Vivek Ramanujan and William Howard-Snyder and Kaifeng Chen and Sham Kakade and Prateek Jain and Ali Farhadi},
    year={2024},
    eprint={2205.13147},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->