export const TraceType = {
  GeneralFunction: "OTHERFUNCTION",
  Gemini: "GEMINI",
  OpenAI: "OPENAI",
  Anthropic: "ANTHROPIC",
  LangChain: "LANGCHAIN",
  LangGraph: "LANGGRAPH",
  LlamaIndex: "LLAMAINDEX",
  GoogleADK: "GOOGLEADK",
  AutoGen: "AUTOGEN",
  AgentToAgent: "AGENTTOAGENT",
  ChromaDB: "CHROMADB",
  Pinecone: "PINECONE",
  Qdrant: "QDRANT",
  Weaviate: "WEAVIATE",
  FAISS: "FAISS",
  Voyage: "VOYAGE",
} as const;

export type TraceTypeValue = (typeof TraceType)[keyof typeof TraceType];

export interface Tokens {
  prompt: number | null;
  completion: number | null;
  total: number | null;
}

export interface LogTrace {
  integration?: TraceTypeValue | null;
  type?: string | null;
  timestamp?: number | null;
  latency?: number | null;
  tokens?: Tokens | null;

  model?: string | null;
  api?: string | null;
  messages?: unknown;
  contents?: unknown;
  input?: unknown;
  system?: unknown;
  system_instruction?: unknown;
  tools?: unknown;
  tool_choice?: unknown;
  tool_config?: unknown;

  response?: unknown;
  thinking?: unknown[] | null;
  tool_calls?: unknown[] | null;
  tool_uses?: unknown[] | null;
  function_calls?: unknown[] | null;
  tool_call_latencies?: unknown;
  finish_reasons?: unknown[] | null;
  stop_reason?: string | null;

  mcp_servers?: unknown[] | null;
  mcp_calls?: unknown[] | null;
  mcp_results?: unknown[] | null;

  trace_id?: string | null;
  parent_id?: string | null;
  function?: string | null;
  output?: unknown;
  success?: boolean | null;

  status?: string | null;
  started_at?: number | null;
  error_traceback?: string | null;

  [key: string]: unknown;
}

export function toPlainObject(trace: LogTrace): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(trace)) {
    if (v !== undefined) result[k] = v;
  }
  return result;
}
