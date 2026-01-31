import warnings
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, root_validator


class TrainingConfig(BaseModel):
    """Configuration for entropy knowledge distillation training."""
    
    # Model settings
    teacher_model: str
    student_model: str
    teacher_quant_bits: Optional[int] = None  # 4 or 8 to enable bitsandbytes quant for teacher
    student_quant_bits: Optional[int] = None  # optional quant for student (usually None for training)
    distill_category: Literal["off_policy", "on_policy"] = "off_policy"
    distill_type: Literal[
        "vanilla",
        "top-k-tok",
        "top-k-tok-dkd",
        "random",
        "random-dkd",
        "bucket",
        "pos-rs-kd",
        "pos-rs-kd-dkd",
        "linucb",
        "linucb-dkd",
        "atkd",
        "dkd",
    ] = "vanilla"
    k_percent: int = Field(default=20, description="for top-k-tok and random")
    atkd_hard_percent: float = Field(
        default=50.0,
        description="Percentage of tokens treated as hard (highest uncertainty) in AT-KD mode.",
    )
    atkd_loss_lambda: float = Field(
        default=0.2,
        description="λ weight applied to easy-token KL in AT-KD (L_all = λL^e + (1-λ)L^h).",
    )
    normalize_topk_by_length: bool = Field(
        default=False,
        description="When true, top-k selection uses a shared quota derived from the batch average length",
    )
    topk_tok_selection_metric: Literal[
        "teacher_entropy",
        "student_entropy",
        "student_ce",
        "kl",
        "reverse-kl",
        "ce_ratio",
        "ce_ratio_entropy",
        "ce_ratio_plus_entropy",
    ] = Field(
        default="teacher_entropy",
        description="Metric used to rank tokens for top-k selection: teacher_entropy (default), student_entropy, student_ce, kl, reverse-kl, ce_ratio (CE_s/CE_t), ce_ratio_entropy (CE_s/CE_t * H_s), or ce_ratio_plus_entropy (CE_s/CE_t + H_s)",
    )
    selection_curriculum: bool = Field(
        default=False,
        description="Enable entropy-driven curriculum that shifts token selection from lower-entropy (easy) to higher-entropy (hard) tokens over training",
    )
    selection_curriculum_steps: int = Field(
        default=2000,
        description="Number of optimization steps used to progress the selection curriculum from start to end target",
    )
    selection_curriculum_start: float = Field(
        default=0.0,
        description="Fraction of the entropy-sorted list to target at the start of the curriculum (0.0 = lowest entropy)",
    )
    selection_curriculum_end: float = Field(
        default=1.0,
        description="Fraction of the entropy-sorted list to target at the end of the curriculum (1.0 = highest entropy)",
    )
    selection_curriculum_power: float = Field(
        default=1.0,
        description="Exponent applied to normalized curriculum progress (values >1 focus longer on easy tokens, <1 accelerates the shift)",
    )
    enable_ce: bool = Field(default=True, description="Enable cross-entropy loss in addition to KD loss")
    enable_ce_on_all_tokens: bool = Field(
        default=False,
        description="When true, apply cross-entropy loss to every valid token even when KD selects a subset",
    )
    alpha_ce: float = Field(
        default=0.1,
        description="Weight for cross-entropy loss (vs KD loss). Total loss = (1-alpha_ce)*L_KD + alpha_ce*L_CE",
    )
    kd_temperature: float = Field(default=2.0, description="Unified KD temperature used for teacher/student log-softmax and loss scaling")
    entropy_approx_temperature: float = Field(default=2.0, description="Temperature used during offline pass for entropy approximation (and RS-KD proposal if applicable)")
    unbounded_to_1_loss: bool = Field(
        default=False,
        description="When true, skip alpha_ce mixing and set total loss = CE + KD (for DKD exact formulation).",
    )
    dkd_alpha: float = Field(default=1.0, description="Weight for DKD target-class binary KL term (TCKD)")
    dkd_beta: float = Field(default=8.0, description="Weight for DKD non-target KL term (NCKD)")
    entropy_cache_approx: bool = Field(
        default=False,
        description="When true, store truncated entropy approximation (entropy_approx cache mode) instead of exact entropy.",
    )
    # KD objective + temperature annealing (optional)
    kd_objective: Literal["forward", "reverse"] = "forward"
    anneal_kd_temperature: bool = Field(default=False, description="Enable annealing schedule for kd_temperature during training")
    kd_temperature_start: float = Field(default=2.0, description="Starting KD temperature when annealing")
    kd_temperature_end: float = Field(default=1.0, description="Final KD temperature when annealing")
    kd_hold_frac: float = Field(default=0.6, description="Fraction of total updates to hold at start temperature before linear decay")
    # RS-KD parameters (for distill_type="pos-rs-kd")
    rs_alpha: float = Field(default=1.0, description="Scale applied to entropy logits before softmax: q(i) ∝ exp(alpha · H_i)")
    rs_epsilon: float = Field(default=0.02, description="Mixture with uniform for tail coverage: q ← (1-ε)q + ε·uniform")
    rs_floor: float = Field(default=1e-6, description="Minimum probability floor to avoid huge weights / degeneracy")
    rs_bucket_mode: bool = Field(
        default=False,
        description="When true, restrict pos-rs-kd sampling to a percentile bucket before sampling",
    )
    rs_bucket_lower_percent: Optional[float] = Field(
        default=None,
        description="Lower percentile bound (0-100) for pos-rs-kd bucket mode",
    )
    rs_bucket_upper_percent: Optional[float] = Field(
        default=None,
        description="Upper percentile bound (0-100) for pos-rs-kd bucket mode",
    )
    pos_rs_entropy_dump_enabled: bool = Field(
        default=True,
        description="When true, dump entropy-derived sampling distributions for the first few documents in pos-rs-kd runs.",
    )
    pos_rs_entropy_dump_limit: int = Field(
        default=3,
        description="Maximum number of documents to record when dumping pos-rs-kd entropy distributions.",
    )
    pos_rs_entropy_dump_path: Optional[str] = Field(
        default=None,
        description="Optional override for the pos-rs-kd entropy dump file (defaults to <output_dir>/pos_rs_entropy_dump.jsonl).",
    )
    pos_rs_match_full_kd: bool = Field(
        default=False,
        description="When true, weight pos-rs-kd samples by 1/q(t) so the estimator matches the per-position full-KD loss in expectation.",
    )
    topk_debug_dump_path: Optional[str] = Field(
        default=None,
        description="Optional JSONL file for logging top-k token selection debug records.",
    )
    topk_debug_dump_limit: int = Field(
        default=0,
        description="Maximum number of documents to dump for top-k token selection debugging (0 disables).",
    )

    # Compute-skip optimizations for selective KD
    cut_after_last_selected: bool = Field(
        default=False,
        description="When true, truncate sequences to (max selected position + 1) before transformer forward, saving compute on trailing unselected tokens.",
    )
    logits_on_selected_only: bool = Field(
        default=False,
        description="When true, only compute lm_head projection on selected token positions (avoids full [B,T,V] logits tensor).",
    )
    teacher_selective_lm_head: bool = Field(
        default=False,
        description="When true, only compute teacher lm_head on positions selected by student entropy (avoids full [B,T,V_teacher] tensor).",
    )
    student_selective_lm_head: bool = Field(
        default=False,
        description="When true, only compute student lm_head (with grad) on selected positions (avoids full [B,T,V_student] forward+backward).",
    )
    selective_lm_head_same_flow: bool = Field(
        default=False,
        description="When true, force the selective lm_head flow even if both selective flags are false (useful for same-flow baselines).",
    )
    entropy_streaming_chunk_size: int = Field(
        default=128,
        description="Number of positions to process at a time during streaming entropy computation (balances memory vs GPU efficiency).",
    )
    log_peak_memory: bool = Field(
        default=False,
        description="When true, log peak GPU memory per step to efficiency CSV and W&B.",
    )
    
    # Bucket mode parameters (for distill_type="bucket")
    bucket_lower_percent: int = Field(default=70, description="Lower bound for bucket mode (e.g., 70% means skip bottom 70%)")
    bucket_upper_percent: int = Field(default=80, description="Upper bound for bucket mode (e.g., 80% means skip top 20%)")

    # Weighted KD: weight each token's KL loss by uncertainty (uses offline_cache_mode to determine weight type)
    weighted_kd: bool = Field(default=False, description="Weight each token's KL by uncertainty (entropy or unc based on offline_cache_mode)")

    weighted_kd_metric: Optional[Literal["entropy", "unc", "student_entropy"]] = Field(
        default=None,
        description=(
            "Metric for weighted-KD token weights. When None, defaults to teacher-entropy weights when "
            "offline_cache_mode is an entropy mode, otherwise uses 'unc'=1-max(p_teacher)."
        ),
    )

    # UDKD (Uncertainty-Driven Decoupled KD) loss
    udkd_loss: bool = Field(default=False, description="Use UDKD loss instead of standard KL divergence")
    udkd_uncertainty_metric: Literal["unc", "entropy", "student_entropy", "kl", "reverse_kl"] = Field(
        default="unc",
        description="UDKD gate metric: 'unc'=1-p(target), 'entropy'=teacher H/log(V), 'student_entropy'=student H/log(V), 'kl'=KL(teacher||student), 'reverse_kl'=KL(student||teacher)",
    )

    # Score-KD parameters
    score_token_selection: bool = Field(default=False, description="Use composite score (entropy + student CE + KL) to rank tokens instead of pure entropy")
    score_normalize: Literal["none", "z", "minmax"] = "z"
    score_entropy_weight: float = Field(default=1.0, description="Weight for teacher entropy component in score-based KD")
    score_ce_weight: float = Field(default=1.0, description="Weight for student cross-entropy component in score-based KD")
    score_kl_weight: float = Field(default=1.0, description="Weight for teacher-student KL component in score-based KD")

    # LinUCB contextual bandit parameters
    bandit_alpha: float = Field(default=1.0, description="Exploration coefficient for LinUCB (higher = more exploratory)")
    bandit_lambda: float = Field(default=1.0, description="L2 regularization for LinUCB covariance matrix")
    bandit_threshold: float = Field(default=0.5, description="Minimum UCB score for a token to be selected")
    bandit_min_tokens: int = Field(default=1, description="Minimum number of tokens to distill per example when using LinUCB")
    bandit_max_tokens: Optional[int] = Field(default=64, description="Optional cap on tokens distilled per example in LinUCB mode")
    bandit_device: str = Field(default="cpu", description="Device to maintain the LinUCB statistics on (cpu or cuda)")
    bandit_reward_clip: float = Field(default=5.0, description="Absolute clip value applied to KL improvement rewards before LinUCB update")
    
    # On-policy distillation settings
    on_policy_max_new_tokens: int = Field(default=256, description="Maximum number of tokens to generate during on-policy rollouts")
    on_policy_temperature: float = Field(default=0.7, description="Sampling temperature used for student rollouts")
    on_policy_top_p: float = Field(default=0.9, description="Top-p nucleus sampling for student rollouts")
    on_policy_do_sample: bool = Field(default=True, description="Enable stochastic sampling when generating student rollouts")
    on_policy_group_size: int = Field(default=1, description="Number of student rollouts per prompt")
    on_policy_reverse_kl_weight: float = Field(default=1.0, description="Weight applied to reverse KL (student||teacher) loss on on-policy tokens")
    on_policy_forward_kl_weight: float = Field(default=0.0, description="Weight applied to forward KL (teacher||student) loss on on-policy tokens")
    on_policy_forward_self_norm: bool = Field(default=True, description="Use self-normalized importance weights for forward KL estimator")
    on_policy_curriculum: bool = Field(default=False, description="Enable curriculum over k_percent during on-policy training")
    on_policy_curriculum_steps: int = Field(default=1000, description="Number of optimization steps to anneal k_percent toward its target")
    on_policy_curriculum_start_k: float = Field(default=5.0, description="Initial k_percent value when curriculum starts (percentage)")
    on_policy_curriculum_power: float = Field(default=1.0, description="Exponent applied to progress when computing curriculum interpolation")
    enable_cuts_in_the_middle_for_on_policy: bool = Field(
        default=True,
        description="When true, sample middle cut-points for on-policy FineWeb rollouts before generation.",
    )
    on_policy_cut_min_tokens: int = Field(
        default=12,
        description="Minimum number of rollout tokens to force after a middle cut when enabled.",
    )
    on_policy_cut_max_tokens: int = Field(
        default=32,
        description="Maximum number of rollout tokens to force after a middle cut when enabled.",
    )
    on_policy_cut_min_context: int = Field(
        default=128,
        description="Minimum prefix length (tokens) to retain before the sampled cut point in on-policy mode.",
    )

    # Dataset settings
    datasets: List[str]
    prompt_col: Optional[str] = None
    answer_col: Optional[str] = None
    dataset_config: Optional[str] = None
    # FineWeb streaming token budget (used when datasets[0] == "fineweb")
    fineweb_tokens: int = Field(default=50_000_000, description="Token budget when streaming FineWeb-Edu")
    enable_packing: bool = Field(
        default=True,
        description="Pack concatenated documents into fixed-length token windows before training",
    )

    # Optional gating of distillation based on a frozen pre-distillation student entropy pass
    skip_by_frozen_student: bool = Field(
        default=False,
        description="When true, run a pre-pass with the initial (frozen) student to compute mean token entropy per sample and distill only on the top l%% highest-entropy samples during training.",
    )
    skip_samples_strategy: Literal["entropy", "kl", "ce_ratio", "random"] = Field(
        default="entropy",
        description=(
            "Strategy used when skip_by_frozen_student is enabled: "
            "'entropy' selects the top-l%% samples by frozen-student mean token entropy (default), "
            "'kl' selects the top-l%% samples by mean token KL divergence between frozen teacher and student, "
            "'ce_ratio' selects the top-l%% samples by mean CE_s/(CE_t+eps) between frozen teacher and student, "
            "'random' selects a random l%% subset of samples (no pre-pass)."
        ),
    )
    L_PERCENT_SAMPLES_TO_KEEP: float = Field(
        default=20.0,
        description="Percentage (0-100) of samples to keep for distillation (top-entropy) when skip_by_frozen_student is enabled (default 20).",
    )
    
    # Training hyperparameters
    epochs: int = 1
    batch_size: int = 128
    gradient_accumulation_steps: int = Field(default=4, description="Number of steps to accumulate gradients")
    max_seq_len: int = Field(default=512, description="Maximum sequence length to save memory")
    lr: float = Field(default=1e-5, description="Learning rate")
    # Reproducibility
    seed: int = Field(default=1337, description="Random seed for reproducibility")
    deterministic: bool = Field(default=False, description="Enable deterministic algorithms (may slow down)")
    
    # Output and logging
    output_dir: str
    tensorboard_dir: str = "tb"
    wandb_project: str = "selective-entropy-knowledge-distillation"
    wandb_entity: str = "selective-entropy-knowledge-distillation"
    wandb_enabled: bool = True
    log_efficiency_csv: bool = Field(
        default=False,
        description="When true, append training efficiency metrics to a CSV table after training completes.",
    )
    efficiency_csv_path: str = Field(
        default="results/table_efficiency_test.csv",
        description="CSV output path for efficiency metrics when log_efficiency_csv is enabled.",
    )
    log_skipping_indices: bool = Field(
        default=False,
        description="When true, append selected skip-sample indices to a JSONL file after the prepass.",
    )
    skipping_indices_path: str = Field(
        default="results/skipping_indices.json",
        description="Output path for append-only skip-sample indices JSONL.",
    )
    # Unified runs registries (per split)
    runs_registry_validation: str = Field(
        default="results/runs_validation.json",
        description="Path to the validation split runs JSON registry",
    )
    runs_registry_test: str = Field(
        default="results/runs_test.json",
        description="Path to the test split runs JSON registry",
    )
    override: bool = Field(default=False, description="If true, run even if an identical-params hash exists in the registry")
    
    # Offline cache (teacher precomputation for entropy approx + RS-KD over vocab)
    offline_cache: bool = True
    offline_cache_dir: Optional[str] = None  # if None, defaults to f"{output_dir}/teacher_offline_cache"
    offline_cache_force_hash: Optional[str] = Field(
        default=None,
        description="Force-use a specific offline cache hash under logits_caches/, ignoring signature mismatches.",
    )
    offline_cache_missing_tolerance: int = Field(
        default=100,
        description="Maximum missing cache items allowed when force-using a cache hash.",
    )
    offline_cache_min_hit_rate: float = Field(
        default=0.9,
        description="Minimum cache hit rate (0-1) required when force-using a cache hash.",
    )

    # Profiler controls (torch.profiler)
    profiler_enabled: bool = Field(default=False, description="Enable torch.profiler tracing during training.")
    profiler_dir: Optional[str] = Field(
        default=None,
        description="Output directory for profiler traces (defaults under results/gpu_util).",
    )
    profiler_wait: int = Field(default=1, description="Profiler schedule: wait steps.")
    profiler_warmup: int = Field(default=1, description="Profiler schedule: warmup steps.")
    profiler_active: int = Field(default=10, description="Profiler schedule: active steps.")
    profiler_repeat: int = Field(default=1, description="Profiler schedule: repeat count.")
    profiler_record_shapes: bool = Field(default=True, description="Record tensor shapes in profiler.")
    profiler_with_stack: bool = Field(default=True, description="Record stack traces in profiler.")
    profiler_profile_memory: bool = Field(default=True, description="Profile memory allocations in profiler.")
    profiler_max_steps: Optional[int] = Field(
        default=200,
        description="Max number of profiler steps to record (None for no cap).",
    )
    offline_cache_mode: Literal["entropy_approx", "entropy", "unc", "none"] = Field(
        default="entropy",
        description="Offline cache mode: entropy_approx (truncated entropy), entropy (exact), unc (store target probabilities), or none (store no teacher uncertainty metric).",
    )
    offline_cache_batch_size: Optional[int] = Field(
        default=None,
        description="Batch size used only when building the offline cache (None or <=0 uses batch_size).",
    )
    # Params used by the offline cache builder
    entropy_approx_m: int = Field(default=20, description="Top-m used in truncated entropy approximation")
    rs_vocab_samples: int = Field(default=64, description="Number of vocab samples per position for RS-KD")
    rs_vocab_beta: float = Field(default=1.0, description="Proposal exponent for RS-KD over vocab: q ∝ p^beta")
    # Entropy cache policy (always stored): True => uint8, False => fp16
    H_hat_u8: bool = Field(default=True, description="Store Ĥ as uint8 (True) or fp16 (False)")
    
    # Global-Level Selection (GLS) over tokens — only affects top-k-tok when enabled
    gls_enabled: bool = Field(default=False, description="Enable global-level selection FIFO queue (only impacts top-k-tok)")
    gls_queue_size: int = Field(default=30000, description="Capacity of GLS FIFO queue for computing global threshold")
    gls_log_threshold: bool = Field(default=False, description="Log the GLS threshold each time it's computed")
    
    # Checkpointing
    checkpoint_steps: int = Field(default=500, description="Save checkpoint every N steps (0 to disable)")
    keep_checkpoints: int = Field(default=3, description="Number of recent checkpoints to keep")
    resume_from_checkpoint: Optional[str] = Field(
        default=None,
        description="Path to a checkpoint (.pt) to resume training from (optional).",
    )
    max_train_hours: Optional[float] = Field(
        default=None,
        description="If set, stop training after this many wall-clock hours (checkpoint + exit).",
    )

    # Distributed training context (derived from torchrun environment)
    ddp_world_size: int = Field(default=1, description="World size for DDP runs")
    ddp_rank: int = Field(default=0, description="Global rank for DDP runs")
    ddp_local_rank: int = Field(default=0, description="Local rank for DDP runs")

    @root_validator(pre=True)
    def _set_offline_cache_mode_for_atkd(cls, values):
        distill_type = values.get("distill_type")
        offline_cache_enabled = bool(values.get("offline_cache", True))
        if distill_type == "atkd" and offline_cache_enabled:
            mode = values.get("offline_cache_mode")
            if mode not in (None, "unc"):
                warnings.warn(
                    "distill_type='atkd' with offline_cache enabled requires offline_cache_mode='unc'; overriding.",
                    RuntimeWarning,
                )
            values["offline_cache_mode"] = "unc"
        return values


class CheckpointData(BaseModel):
    """Structure for training checkpoint data."""
    
    epoch: int
    step: int
    global_step: int
    distill_type: str
    k_percent: int
    model_state_dict: dict
    optimizer_state_dict: dict
    
    class Config:
        # Allow arbitrary types for PyTorch state dicts
        arbitrary_types_allowed = True


class TrainingMetrics(BaseModel):
    """Structure for training metrics and logging."""
    
    loss: float
    kl_loss: float
    ce_loss: float
    epoch: int
    step: int
    global_step: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "train/loss": self.loss,
            "train/kl_loss": self.kl_loss,
            "train/ce_loss": self.ce_loss,
            "train/epoch": self.epoch
        }
    
    def to_wandb_dict(self) -> dict:
        """Convert to W&B-specific dictionary with additional context."""
        return {
            "train/loss": self.loss,
            "train/kl_loss": self.kl_loss,
            "train/ce_loss": self.ce_loss,
            "train/epoch": self.epoch,
            "train/step": self.step,
            "train/global_step": self.global_step,
        }
    
    def to_running_dict(self) -> dict:
        """Convert to running averages dictionary."""
        return {
            "loss": self.loss,
            "kl": self.kl_loss,
            "ce": self.ce_loss
        }
