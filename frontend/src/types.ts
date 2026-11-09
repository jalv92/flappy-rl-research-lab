export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
export type Difficulty = 'easy' | 'normal' | 'hard';
export type Mode = 'human' | 'ai' | 'human_vs_ai' | 'ai_vs_ai';
export interface WorldConfig {
  tick_rate: number; world_height: number; view_width: number; bird_x: number; bird_radius: number;
  start_y: number; gravity: number; flap_velocity: number; max_fall_speed: number; speed: number;
  obstacle_width: number; obstacle_spacing: number; first_obstacle_x: number; gap_min: number;
  gap_max: number; gap_margin: number; max_gap_shift: number;
}
export interface GameConfig {
  engine_version: string; modes: Record<Mode, string[]>; difficulties: Record<Difficulty, WorldConfig>;
  trained_on: string; observation: { names: string[]; dim: number; version: string };
  actions: Record<string, string>; speed_range: [number, number]; view_width_range?: [number, number]; default_limit_ticks: number; max_limit_ticks: number;
}
export interface PlayerInfo {
  kind: 'human' | 'ai'; label: string; model_id: string | null; algo?: string; variant?: string | null;
  status?: string; train_steps?: number; diagnostic?: 'q_values' | 'action_probs';
}
export interface Diagnostic { obs: number[]; action: number; q_values?: number[]; action_probs?: number[] }
export interface PlayerState { y: number; vy: number; alive: boolean; score: number; distance: number; flapped: boolean; death: string | null; diag?: Diagnostic }
export interface ReadyMessage {
  type: 'ready'; mode: Mode; seed: number; difficulty: Difficulty; in_distribution: boolean;
  limit_ticks: number; replay_of: number | null; personal_best: number | null; config: WorldConfig; players: PlayerInfo[];
}
export interface Snapshot {
  type: 'state'; phase: 'ready' | 'running' | 'paused' | 'over'; tick: number; time: number; speed: number; scroll_x: number;
  obstacles: { i: number; x: number; gap_top: number; gap_bottom: number }[]; players: PlayerState[];
}
export interface MatchResult {
  match_id: number; mode: Mode; seed: number; difficulty: Difficulty; ticks: number; limit_reached: boolean;
  winner: number | null; draw: boolean; personal_best?: number | null; new_record?: boolean; replay?: boolean;
  players: (PlayerInfo & { score: number; distance: number; ticks: number; flaps: number; death: string | null; died_tick: number | null })[];
}
export type ServerMessage = ReadyMessage | Snapshot | { type: 'game_over'; result: MatchResult } | { type: 'error'; message: string } | { type: 'stopped' };
export type GameCommand = { type: 'start'; mode: Mode; models: string[]; seed?: number; difficulty: Difficulty; speed?: number; limit_ticks?: number; view_width?: number }
  | { type: 'flap' | 'pause' | 'resume' | 'stop' } | { type: 'restart'; same_seed?: boolean; seed?: number }
  | { type: 'set_speed'; speed: number } | { type: 'replay'; match_id: number; speed?: number; view_width?: number };
export interface MatchRow { id: number; mode: Mode; seed: number; difficulty: Difficulty; participants: PlayerInfo[]; result: MatchResult; created_at: number; replay?: Json }
export interface MatchStats { human: { games: number; best: Record<string, number>; vs_ai: { wins: number; losses: number; draws: number } }; head_to_head: { a: string; b: string; wins_a: number; wins_b: number; draws: number; n: number }[]; total: number }
export interface Series { model_a: string; model_b: string; n: number; wins_a: number; wins_b: number; draws: number; mean_score_a: number; mean_score_b: number; matches: { match_id: number; seed: number; winner: number | null; draw: boolean; scores: number[]; ticks: number[] }[] }
export interface ModelMetrics {
  final_val?: { n: number; mean: number; median: number; max: number; min: number; std: number; success: Record<string, number> };
  best_val_mean?: number; best_step?: number; steps_to?: Record<string, number | null>;
  auc?: number; wall_seconds?: number; steps?: number; reason?: string;
}
export interface Model {
  id: string; run_id: number; algo: string; variant: string | null; status: string; seed: number;
  experiment_id: number | null; train_steps: number; code_version: string | null; env_version: string;
  created_at: number; updated_at: number; config: Json; metrics: ModelMetrics | null; obs_dim: number; obs_version: string; n_actions: number;
  checkpoint_exists?: boolean; evals?: Evaluation[];
}
export interface Evaluation { id: number; purpose: string; split: string; scenario: string; step: number; n: number; mean: number; median: number; max: number; min: number; std: number; success: Record<string, number>; created_at: number }
export interface Run {
  id: number; kind: string; algo: string; variant: string | null; seed: number; profile: string; experiment_id: number | null;
  arm: string; status: string; stop_requested: number; steps: number; total_steps: number; created_at: number;
  started_at: number | null; finished_at: number | null; wall_seconds: number; code_version: string | null;
  env_version: string; device: string | null; hardware: Json; model_id: string | null; error: string | null; config: Json;
  metric_keys?: string[]; summary?: { curve: [number, number][]; final_mean: number | null; best_mean: number | null; auc: number | null; sps: number | null; wall_seconds: number; model_id: string | null };
}
export type Metrics = Record<string, [number, number][]>;
export interface Algorithms { algorithms: { id: string; name: string; kind: string; supports_variants: boolean; default_hyperparameters: Json }[]; variants: { id: string; is_control: boolean; hypothesis: Json }[]; variant_composition: string }
export interface Profiles { limits: Record<string, number>; profiles: Record<string, { description: string; steps: number; seeds: number; eval_every: number; eval_episodes: number; max_experiments: number; session_wall_seconds: number; confirm: boolean }>; overridable: Record<string, { min: number; max: number }> }
export interface Benchmark { env_version: string; total_steps: number; runs?: number; groups?: number; validation?: string }
export interface Benchmarks { benchmarks: Benchmark[]; metrics: Record<string, string> }
export interface Estimate { mean?: number | null; ci?: number[] | Record<string, never>; std?: number; values?: number[] }
export interface CompareGroup {
  group_key: string; label: string; algo: string; variant: string | null; kind: string; total_steps: number; env_version: string;
  n_seeds: number; seeds: number[]; run_ids: number[]; final: Estimate; best: Estimate; auc: Estimate;
  wall_seconds_mean: number | null; sps_mean: number | null;
  curve: { step: number; mean: number; lo: number; hi: number; n: number }[];
  generalization: Record<string, Estimate>;
}
export interface Comparison { benchmark: Benchmark; groups: CompareGroup[] }
export interface LeaderRow { rank?: number; label: string; group_key: string; algo: string; variant: string | null; kind: string; n_seeds: number; value: number | null; ci: number[] | Record<string, never> | null; final_mean: number | null; wall_seconds_mean: number | null; run_ids: number[] }
export interface Leaderboard { metric: string; description: string; lower_is_better: boolean; benchmark: Benchmark; rows: LeaderRow[] }
export interface Pair { a: string; b: string; final_mean: PairEstimate; auc: PairEstimate; wall_ratio: number | null }
export interface PairEstimate { mean_a?: number; mean_b?: number; diff?: number; ci?: number[]; rel_diff?: number | null; prob_improvement?: number; n_a?: number; n_b?: number }
export interface Hypothesis { id: string; title: string; area: string; variant: string; targets: string; motivation: string; modification: string; expected_benefit: string; failure_modes: string; criteria: string; baseline: string; prior_work: string; novelty: string; budget: Json; status: string; source: string }
export interface Experiment { id: number; session_id: number; hypothesis_id: string; hypothesis_title?: string; variant?: string; stage: string; profile: string; status: string; decision: string | null; conclusion: string | null; code_version: string | null; started_at: number; finished_at: number | null; error: string | null; config: Json; result: Json; hypothesis?: Hypothesis; runs?: Run[]; report_markdown?: string | null }
export interface Session { id: number; profile: string; objective: string; budgets: Record<string, Json>; status: string; control: string; experiments_done: number; notes: string | null; started_at: number; finished_at: number | null; error: string | null; experiments?: Experiment[] }
export interface QueueItem { hypothesis_id: string; title: string; variant: string; stage: string; reason: string }
export interface Observations { weaknesses: { name: string; severity: number; evidence: string }[]; evidence: Json; note: string | null }
export interface LabEvent { id: number; t: number; level: string; source: string; run_id: number | null; message: string }
export interface Status {
  time: number; research: { status: string; session: Session | null; active_experiment: Experiment | null };
  training: { active_runs: Pick<Run, 'id' | 'algo' | 'variant' | 'seed' | 'status' | 'steps' | 'total_steps' | 'experiment_id' | 'arm'>[]; max_concurrent: number };
  totals: { experiments_completed: number; experiments_failed: number; runs_completed: number; training_steps: number; models: number; matches: number; held_out_test_uses: number };
  best_verified_model: { id: string; algo: string; variant: string | null; train_steps: number; status: string; evidence: Json } | null;
  best_score: number | null; resources: { cpu_percent: number; cpu_count: number; ram_used_mb: number; ram_total_mb: number; disk_used_mb: number; disk_limit_mb: number; gpu: { name: string; utilization_percent: number | null; memory_used_mb: number; memory_total_mb: number } | null; device_benchmark: Json }; limits: Record<string, number>;
}
