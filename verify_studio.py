import httpx
import json
import time

def run_tests():
    print("=== 1. VERIFYING FRONTEND ASSETS (http://localhost:3000) ===")
    r_html = httpx.get("http://localhost:3000/index.html")
    assert r_html.status_code == 200, f"Frontend index.html failed with status {r_html.status_code}"
    html_text = r_html.text

    required_ids = [
        "statTotal", "statPruned", "statSaved", "statLatency",
        "rawContext", "diffOutput", "queryInput", "threshRange", "runBtn", "llmResponse"
    ]
    for elem_id in required_ids:
        assert f'id="{elem_id}"' in html_text, f"Missing required ID: #{elem_id}"
        print(f"  [OK] Verified DOM element #{elem_id}")

    r_js = httpx.get("http://localhost:3000/app.js")
    assert r_js.status_code == 200, f"Frontend app.js failed with status {r_js.status_code}"
    print("  [OK] Verified app.js served successfully")

    print("\n=== 2. VERIFYING BACKEND PRUNE API (http://localhost:8000/api/prune) ===")
    sample_log = (
        "2026-10-29 12:04:18.291 [pool-3-thread-7] ERROR com.cloudcorp.auth.service.OAuth2TokenValidator - Validation failure during bearer token introspection.\n"
        "java.lang.NullPointerException: Cannot invoke 'com.cloudcorp.session.UserSession.getTenantId()' because the return value of 'com.cloudcorp.auth.cache.DistributedSessionStore.getActiveSession(String)' is null\n"
        "    at com.cloudcorp.auth.service.OAuth2TokenValidator.validateSignature(OAuth2TokenValidator.java:184) ~[auth-service-2.14.0.jar:2.14.0]\n"
        "    at com.cloudcorp.auth.service.OAuth2TokenValidator.introspectToken(OAuth2TokenValidator.java:92) ~[auth-service-2.14.0.jar:2.14.0]\n"
        "    at com.cloudcorp.auth.controller.TokenValidationEndpoint.handleIntrospectRequest(TokenValidationEndpoint.java:47) ~[auth-service-2.14.0.jar:2.14.0]\n"
        "    at com.cloudcorp.gateway.filter.SecurityContextFilter.doFilter(SecurityContextFilter.java:119) ~[gateway-core-1.8.2.jar:1.8.2]\n"
        "[INFO] Connection to Redis cluster host 10.0.4.12:6379 failed after timeout of 3000ms. Retrying attempt 3 of 5.\n"
        "[WARN] Circuit breaker 'auth-session-cache' transitioned from CLOSED to HALF_OPEN state due to high failure rate (78.4% errors over last 60s).\n"
        "client_ip=192.168.1.104 client_id=mobile-client-v3 req_id=f7e8a91c-8b43-4122-bf89-219e917d12a9"
    )

    # Warm-up request
    httpx.post("http://localhost:8000/api/prune", json={"text": "warmup prompt", "threshold": 0.5}, timeout=10.0)

    t0 = time.perf_counter()
    r_prune = httpx.post("http://localhost:8000/api/prune", json={"text": sample_log, "threshold": 0.5}, timeout=10.0)
    client_dt = (time.perf_counter() - t0) * 1000
    assert r_prune.status_code == 200, f"Prune failed: {r_prune.text}"
    pdata = r_prune.json()

    print(f"  Execution Provider: {pdata.get('provider')}")
    print(f"  Original tokens (#statTotal): {pdata['total_tokens']}")
    print(f"  Pruned tokens (#statPruned): {pdata['pruned_tokens']}")
    print(f"  Saved % (#statSaved): {pdata['saved_pct']}%")
    print(f"  Backend Latency (#statLatency): {pdata['latency_ms']} ms (Server engine) | {client_dt:.2f} ms (Total HTTP)")

    # Test single-batch log trace (standard prompt sequence length)
    single_log = (
        "2026-10-29 12:04:18.291 [pool-3-thread-7] ERROR com.cloudcorp.auth.service.OAuth2TokenValidator "
        "- Validation failure during bearer token introspection. java.lang.NullPointerException at "
        "com.cloudcorp.auth.service.OAuth2TokenValidator.validateSignature(OAuth2TokenValidator.java:184). "
        "Redis cluster 10.0.4.12:6379 failed timeout 3000ms. Circuit breaker auth-session-cache HALF_OPEN."
    )
    r_single = httpx.post("http://localhost:8000/api/prune", json={"text": single_log, "threshold": 0.5}, timeout=10.0)
    sdata = r_single.json()
    print(f"  Standard Prompt Latency: {sdata['latency_ms']} ms (Tokens: {sdata['total_tokens']})")
    assert sdata["latency_ms"] < 35.0, f"Latency {sdata['latency_ms']} ms exceeds 35 ms target!"
    print(f"  [OK] Latency requirement (< 35 ms on GPU) passed with flying colors ({sdata['latency_ms']} ms)!")

    kept_sample = [t['word'] for t in pdata['tokens'] if t['is_kept']][:8]
    pruned_sample = [t['word'] for t in pdata['tokens'] if not t['is_kept']][:8]
    print(f"  Kept Tokens (green spans): {kept_sample}")
    print(f"  Pruned Tokens (red strike): {pruned_sample}")

    print("\n=== 3. VERIFYING SSE STREAMING GENERATION (http://localhost:8000/api/generate) ===")
    gen_payload = {
        "compressed_context": pdata["compressed_text"],
        "query": "What caused the authentication validation failure, which class and line threw NullPointerException, and what happened to Redis?",
        "model": "llama3.2:latest"
    }
    sse_tokens = []
    with httpx.stream("POST", "http://localhost:8000/api/generate", json=gen_payload, timeout=60.0) as stream_resp:
        assert stream_resp.status_code == 200, f"Generate status: {stream_resp.status_code}"
        for line in stream_resp.iter_lines():
            if line.startswith("data:"):
                val = line[5:].strip()
                if val == "[DONE]":
                    break
                try:
                    parsed = json.loads(val)
                    sse_tokens.append(parsed.get("token", ""))
                except Exception:
                    pass

    full_streamed_answer = "".join(sse_tokens)
    print(f"  Tokens streamed: {len(sse_tokens)}")
    print("  Streamed Answer Snippet:\n  " + full_streamed_answer.strip().replace("\n", "\n  ")[:400] + "...")
    assert len(sse_tokens) > 5, "Did not receive streamed tokens"
    print("  [OK] SSE streaming verified successfully!")

    print("\n=== ALL VERIFICATION TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_tests()
