import {
  patchOpenAI,
  patchOpenAIResponses,
  patchOpenAIParse,
  patchOpenAIStreamHelper,
  patchOpenAIEmbeddings,
  patchOpenAIImages,
  patchOpenAIAudio,
} from "./openai";
import { patchAnthropic, patchAnthropicBeta } from "./anthropic";
import {
  patchGemini,
  patchGeminiCountTokens,
  patchGeminiEmbeddings,
  patchGeminiVertex,
} from "./gemini";
import { patchLangchain, patchLangGraph } from "./langchain";
import { patchGoogleADK } from "./googleadk";
import { patchChromaDB } from "./vectorstores/chromadb";
import { patchPinecone } from "./vectorstores/pinecone";
import { patchQdrant } from "./vectorstores/qdrant";
import { patchWeaviate } from "./vectorstores/weaviate";
import { patchFAISS } from "./vectorstores/faiss";
import { patchVoyage } from "./voyage";
import { patchMcpInitialize, patchMcpListTools, patchMcpCallTool } from "./shared/mcpPatch";

function _safe(fn: () => void): void {
  try {
    fn();
  } catch {
    // Silently skip if the SDK is not installed
  }
}

let _initialized = false;

export function initIntegrations(): void {
  if (_initialized) return;
  _initialized = true;

  _safe(patchOpenAI);
  _safe(patchOpenAIResponses);
  _safe(patchOpenAIParse);
  _safe(patchOpenAIStreamHelper);
  _safe(patchOpenAIEmbeddings);
  _safe(patchOpenAIImages);
  _safe(patchOpenAIAudio);
  _safe(patchAnthropic);
  _safe(patchAnthropicBeta);
  _safe(patchGemini);
  _safe(patchGeminiCountTokens);
  _safe(patchGeminiEmbeddings);
  _safe(patchGeminiVertex);
  _safe(patchLangchain);
  _safe(patchLangGraph);
  _safe(patchGoogleADK);
  _safe(patchChromaDB);
  _safe(patchPinecone);
  _safe(patchQdrant);
  _safe(patchWeaviate);
  _safe(patchFAISS);
  _safe(patchVoyage);
  _safe(patchMcpInitialize);
  _safe(patchMcpListTools);
  _safe(patchMcpCallTool);
}
