// Shared types mirroring the backend event + resource schemas.

export interface RunConfig {
  model: string;
  base_url?: string;
  gen_ai_system?: string;
  temperature: number;
  top_p: number;
  max_tokens: number;
  seed: number | null;
  max_steps: number;
  search_top_k: number;
  max_tool_workers: number;
  elicit_reasoning: boolean;
  enable_sub_agent: boolean;
  sub_agent_max_steps: number;
}

export interface ToolCall {
  id: string;
  name: string;
  arguments: string;
}

export interface Completion {
  content: string;
  tool_calls: ToolCall[];
}

// A normalized span event (one per finished LLM/tool/root span).
export interface SpanEvent {
  kind: "span";
  node_id: string;
  node_type: "agent_run" | "llm_call" | "tool_call";
  name: string;
  step: number | null;
  depends_on: string[];
  start_time_ns: number | null;
  end_time_ns: number | null;
  duration_ns: number | null;
  is_nested: boolean;
  model?: string | null;
  temperature?: number | null;
  top_p?: number | null;
  seed?: number | null;
  input_tokens: number;
  output_tokens: number;
  finish_reasons?: string[] | null;
  tool_call_count?: number | null;
  prompt?: string | null;
  completion?: Completion | null;
  tool_name?: string | null;
  tool_args?: Record<string, any>;
  tool_result?: any;
  outcome?: string | null;
  answer?: string | null;
  question?: string | null;
  error?: string | null;
}

export interface GraphNode {
  id: string;
  type: string;
  label: string;
  name?: string;
  step: number | null;
  level: number;
  tool_name?: string | null;
  tool_args?: Record<string, any>;
  input_tokens: number;
  output_tokens: number;
  duration_ns: number;
  start_time_ns?: number | null;
  end_time_ns?: number | null;
  outcome?: string | null;
  answer?: string | null;
  question?: string | null;
  error?: string | null;
  is_repeat: boolean;
  repeat_labels: string[];
  is_nested: boolean;
}

export interface GraphEdge {
  source: string;
  target: string;
}

export interface BehavioralRepeat {
  kind: string;
  from_node: string;
  to_node: string;
  tool_name: string;
  detail: string;
}

export interface GraphJSON {
  nodes: GraphNode[];
  edges: GraphEdge[];
  levels: Record<string, number>;
  metrics: Record<string, any>;
  behavioral_repeats: BehavioralRepeat[];
}

export interface Report {
  wall_clock_s: number;
  stage_times: { llm_s: number; tool_s: number; overhead_s: number };
  num_llm_calls: number;
  num_tool_calls: number;
  num_searches: number;
  num_reads: number;
  num_sub_agents: number;
  reasoning_iterations: number;
  node_count: number;
  edge_count: number;
  depth: number;
  width: number;
  width_executed: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  tool_breakdown: Record<string, number>;
  cost: { usd: number; note: string; [k: string]: any };
  memory: any;
}

export interface RunFinished {
  kind: "run_finished";
  run_id: string;
  task_id: string;
  question: string;
  answer: string | null;
  outcome: string;
  graded: boolean;
  config?: RunConfig;
  graph: GraphJSON;
  report: Report;
  trace_file?: string;
  trace_id?: string;
  mode: "live" | "replay";
  created_at: number;
}

export interface RunStarted {
  kind: "run_started";
  run_id: string;
  task_id: string;
  question: string;
  graded: boolean;
  mode?: string;
  config?: RunConfig;
}

export interface RunError {
  kind: "error";
  run_id: string;
  error: string;
}

export type StreamEvent = SpanEvent | RunFinished | RunStarted | RunError;

export interface ModelInfo {
  served: string;
  label: string;
  system?: string;
  parser?: string | null;
  source?: string;
  available?: boolean;
}

export interface PromptPreset {
  task_id: string;
  question: string;
  answers: string[];
  hops: number;
  supporting: string[];
  group: string;
  branch: boolean;
}

export interface Doc {
  id: string;
  title: string;
  text: string;
}

export interface TraceInfo {
  file: string;
  name: string;
  num_spans: number;
  num_runs: number;
  tasks: string[];
  size_bytes: number;
}

export interface HistoryEntry {
  id: string;
  run_id: string;
  timestamp: number;
  mode: string;
  task_id: string;
  question: string;
  model: string;
  outcome: string;
  answer_preview: string;
  node_count: number;
  total_tokens: number;
  wall_clock_s: number;
  config: RunConfig;
}
