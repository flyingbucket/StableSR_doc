# Class: DDPM
## Inheritance Tree (MRO):
- DDPM
- LightningModule
- ABC
- DeviceDtypeModuleMixin
- HyperparametersMixin
- GradInformation
- ModelIO
- ModelHooks
- DataHooks
- CheckpointHooks
- Module
- object

## Instance Attributes (from self.xxx assignments):
- parameterization (defined in: [[__init__]])
- cond_stage_model (defined in: [[__init__]])
- clip_denoised (defined in: [[__init__]])
- log_every_t (defined in: [[__init__]])
- first_stage_key (defined in: [[__init__]])
- image_size (defined in: [[__init__]])
- channels (defined in: [[__init__]])
- use_positional_encodings (defined in: [[__init__]])
- model (defined in: [[__init__]])
- use_ema (defined in: [[__init__]])
- use_scheduler (defined in: [[__init__]])
- v_posterior (defined in: [[__init__]])
- original_elbo_weight (defined in: [[__init__]])
- l_simple_weight (defined in: [[__init__]])
- loss_type (defined in: [[__init__]])
- learn_logvar (defined in: [[__init__]])
- logvar (defined in: [[__init__, __init__]])
- model_ema (defined in: [[__init__]])
- scheduler_config (defined in: [[__init__]])
- monitor (defined in: [[__init__]])
- num_timesteps (defined in: [[register_schedule]])
- linear_start (defined in: [[register_schedule]])
- linear_end (defined in: [[register_schedule]])

## Project-defined Methods:
- [[__init__]]  ←  [[DDPM]]
- [[_get_rows_from_list]]  ←  [[DDPM]]
- [[configure_optimizers]]  ←  [[DDPM]]
- [[forward]]  ←  [[DDPM]]
- [[get_input]]  ←  [[DDPM]]
- [[get_loss]]  ←  [[DDPM]]
- [[get_v]]  ←  [[DDPM]]
- [[init_from_ckpt]]  ←  [[DDPM]]
- [[on_train_batch_end]]  ←  [[DDPM]]
- [[p_losses]]  ←  [[DDPM]]
- [[p_mean_variance]]  ←  [[DDPM]]
- [[predict_start_from_noise]]  ←  [[DDPM]]
- [[predict_start_from_z_and_v]]  ←  [[DDPM]]
- [[q_mean_variance]]  ←  [[DDPM]]
- [[q_posterior]]  ←  [[DDPM]]
- [[q_sample]]  ←  [[DDPM]]
- [[q_sample_respace]]  ←  [[DDPM]]
- [[register_schedule]]  ←  [[DDPM]]
- [[shared_step]]  ←  [[DDPM]]
- [[training_step]]  ←  [[DDPM]]

## Project-defined Attributes: