// The only file that talks to the backend: plain fetch + JSON (FRONTEND_GUIDE section 1).
import type { Example, Health, ResultsPayload, SearchResponse, StatuteResponse } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, body?: object): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      method: body ? "POST" : "GET",
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new Error(`Cannot reach the CaseTrail API at ${API_URL}. Start it with: uvicorn api.main:app`);
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(`${data.error ?? response.status}: ${data.detail ?? response.statusText}`);
  }
  return data as T;
}

export function getHealth(): Promise<Health> {
  return request<Health>("/health");
}

export function searchPrecedents(text: string, system: string, k = 10): Promise<SearchResponse> {
  return request<SearchResponse>("/search", { text, system, k });
}

export function predictStatutes(text: string, method: string, top = 5): Promise<StatuteResponse> {
  return request<StatuteResponse>("/statutes", { text, method, top });
}

export async function getExamples(limit = 8): Promise<Example[]> {
  const data = await request<{ examples: Example[] }>(`/examples?limit=${limit}`);
  return data.examples;
}

export function getResults(): Promise<ResultsPayload> {
  return request<ResultsPayload>("/results");
}
