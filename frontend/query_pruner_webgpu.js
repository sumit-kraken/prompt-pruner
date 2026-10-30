/**
 * Query-Conditioned Pruning on WebGPU (Phase 2)
 *
 * Implements a custom WebGPU WGSL Compute Shader that conditions LLMLingua-2
 * token retention scores on the user's query embeddings via cross-attention.
 *
 * Mathematical formulation:
 *   S_base(c_i) = LLMLingua-2 raw probability of token c_i
 *   Attn(c_i, Q) = max_j cosine_similarity(e(c_i), e(q_j))
 *   S_conditioned(c_i) = alpha * S_base(c_i) + (1 - alpha) * Attn(c_i, Q)
 */

export class WebGPUQueryPruner {
  constructor(embedDim = 64) {
    this.embedDim = embedDim;
    this.device = null;
    this.pipeline = null;
    this.shaderModule = null;
    this.isSupported = false;
    this.initialized = false;
  }

  async init() {
    if (this.initialized) return this.isSupported;

    if (!navigator.gpu) {
      console.warn("[WebGPUQueryPruner] WebGPU not supported on this browser.");
      this.isSupported = false;
      this.initialized = true;
      return false;
    }

    try {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance",
      });
      if (!adapter) {
        this.isSupported = false;
        this.initialized = true;
        return false;
      }

      this.device = await adapter.requestDevice();

      // Compile WGSL Cross-Attention Compute Shader
      const wgslCode = `
        struct ShaderParams {
          num_context_tokens: u32,
          num_query_tokens: u32,
          embed_dim: u32,
          alpha: f32,
          temperature: f32,
        };

        @group(0) @binding(0) var<storage, read> context_embeds: array<f32>;
        @group(0) @binding(1) var<storage, read> query_embeds: array<f32>;
        @group(0) @binding(2) var<storage, read> base_scores: array<f32>;
        @group(0) @binding(3) var<storage, read_write> out_scores: array<f32>;
        @group(0) @binding(4) var<storage, read_write> out_attn: array<f32>;
        @group(0) @binding(5) var<uniform> params: ShaderParams;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let i = global_id.x;
          if (i >= params.num_context_tokens) {
            return;
          }

          var max_sim: f32 = -1.0;
          let dim = params.embed_dim;

          // Cross-attention dot-product against all query tokens
          for (var q = 0u; q < params.num_query_tokens; q = q + 1u) {
            var dot_val: f32 = 0.0;
            var norm_c: f32 = 0.0;
            var norm_q: f32 = 0.0;

            for (var d = 0u; d < dim; d = d + 1u) {
              let c = context_embeds[i * dim + d];
              let q_vec = query_embeds[q * dim + d];
              dot_val = dot_val + (c * q_vec);
              norm_c = norm_c + (c * c);
              norm_q = norm_q + (q_vec * q_vec);
            }

            let denom = sqrt(norm_c) * sqrt(norm_q);
            var sim: f32 = 0.0;
            if (denom > 1e-6) {
              sim = dot_val / denom;
            }

            if (sim > max_sim) {
              max_sim = sim;
            }
          }

          // Scaled query relevance score in [0.0, 1.0]
          let query_relevance = clamp((max_sim + 1.0) * 0.5, 0.0, 1.0);
          out_attn[i] = query_relevance;

          // Modulate baseline score:
          // alpha = weight for baseline LLMLingua-2
          // (1 - alpha) = weight for query cross-attention
          let base = base_scores[i];
          let conditioned = (params.alpha * base) + ((1.0 - params.alpha) * query_relevance);

          out_scores[i] = clamp(conditioned, 0.0, 1.0);
        }
      `;

      this.shaderModule = this.device.createShaderModule({
        label: "CrossAttentionShader",
        code: wgslCode,
      });

      this.pipeline = this.device.createComputePipeline({
        label: "CrossAttentionPipeline",
        layout: "auto",
        compute: {
          module: this.shaderModule,
          entryPoint: "main",
        },
      });

      this.isSupported = true;
      this.initialized = true;
      console.log("[WebGPUQueryPruner] WebGPU Cross-Attention Compute Pipeline initialized successfully.");
      return true;
    } catch (e) {
      console.error("[WebGPUQueryPruner] Init failed:", e);
      this.isSupported = false;
      this.initialized = true;
      return false;
    }
  }

  /**
   * Fast semantic subword / character-ngram feature embedding generator.
   * Maps word tokens to dense 64-dimensional feature vectors.
   */
  embedToken(text, dim = 64) {
    const vec = new Float32Array(dim);
    const clean = text.toLowerCase().replace(/[^a-z0-9_.]/g, "");
    if (!clean) return vec;

    // Trigram frequency hashing with positional encoding
    for (let i = 0; i < clean.length - 1; i++) {
      const gram = clean.substring(i, Math.min(i + 3, clean.length));
      let hash = 0;
      for (let j = 0; j < gram.length; j++) {
        hash = (hash << 5) - hash + gram.charCodeAt(j);
        hash |= 0;
      }
      const idx1 = Math.abs(hash) % dim;
      const idx2 = Math.abs(hash * 31 + 17) % dim;
      vec[idx1] += Math.cos(i + 1);
      vec[idx2] += Math.sin(i + 1);
    }

    // Exact word hashing anchor
    let wordHash = 0;
    for (let j = 0; j < clean.length; j++) {
      wordHash = (wordHash << 5) - wordHash + clean.charCodeAt(j);
      wordHash |= 0;
    }
    const anchorIdx = Math.abs(wordHash) % dim;
    vec[anchorIdx] += 2.0;

    // L2 Normalize
    let norm = 0;
    for (let d = 0; d < dim; d++) norm += vec[d] * vec[d];
    norm = Math.sqrt(norm);
    if (norm > 1e-6) {
      for (let d = 0; d < dim; d++) vec[d] /= norm;
    }
    return vec;
  }

  embedBatch(tokens, dim = 64) {
    const total = tokens.length;
    const buffer = new Float32Array(total * dim);
    for (let i = 0; i < total; i++) {
      const v = this.embedToken(tokens[i], dim);
      buffer.set(v, i * dim);
    }
    return buffer;
  }

  /**
   * Run query-conditioned cross-attention on WebGPU.
   *
   * @param {Array<string>} contextTokens - List of context token strings
   * @param {Array<number>} baseScores - Original LLMLingua-2 scores (0.0 - 1.0)
   * @param {string} queryString - User prompt query
   * @param {number} alpha - Base score weight (0.0 = full query, 1.0 = full base, default 0.5)
   * @returns {Promise<{ scores: Float32Array, attention: Float32Array, gpuLatencyMs: number }>}
   */
  async conditionScores(contextTokens, baseScores, queryString, alpha = 0.5) {
    const numContext = contextTokens.length;
    if (numContext === 0) {
      return { scores: new Float32Array(0), attention: new Float32Array(0), gpuLatencyMs: 0 };
    }

    // Tokenize query into semantic words
    const queryWords = queryString
      .split(/\s+/)
      .map(w => w.trim())
      .filter(w => w.length > 1);

    if (queryWords.length === 0) {
      return {
        scores: new Float32Array(baseScores),
        attention: new Float32Array(numContext).fill(0.5),
        gpuLatencyMs: 0,
      };
    }

    const numQuery = queryWords.length;
    const dim = this.embedDim;

    // Ensure WebGPU is ready
    const ready = await this.init();

    if (!ready || !this.device) {
      // Fallback to fast CPU JavaScript cross-attention
      return this.cpuFallback(contextTokens, baseScores, queryWords, alpha);
    }

    const t0 = performance.now();

    // 1. Prepare Host Buffers
    const contextEmbedsData = this.embedBatch(contextTokens, dim);
    const queryEmbedsData = this.embedBatch(queryWords, dim);
    const baseScoresData = new Float32Array(baseScores);

    const uniformData = new ArrayBuffer(20);
    const u32View = new Uint32Array(uniformData, 0, 3);
    const f32View = new Float32Array(uniformData, 12, 2);
    u32View[0] = numContext;
    u32View[1] = numQuery;
    u32View[2] = dim;
    f32View[0] = alpha;
    f32View[1] = 1.0; // Temperature

    // 2. Create GPU Buffers
    const contextBuf = this.device.createBuffer({
      size: contextEmbedsData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(contextBuf, 0, contextEmbedsData);

    const queryBuf = this.device.createBuffer({
      size: queryEmbedsData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(queryBuf, 0, queryEmbedsData);

    const baseScoresBuf = this.device.createBuffer({
      size: baseScoresData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(baseScoresBuf, 0, baseScoresData);

    const outScoresBuf = this.device.createBuffer({
      size: numContext * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const outAttnBuf = this.device.createBuffer({
      size: numContext * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const uniformBuf = this.device.createBuffer({
      size: uniformData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(uniformBuf, 0, uniformData);

    const readbackScoresBuf = this.device.createBuffer({
      size: numContext * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const readbackAttnBuf = this.device.createBuffer({
      size: numContext * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // 3. Create Bind Group
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: contextBuf } },
        { binding: 1, resource: { buffer: queryBuf } },
        { binding: 2, resource: { buffer: baseScoresBuf } },
        { binding: 3, resource: { buffer: outScoresBuf } },
        { binding: 4, resource: { buffer: outAttnBuf } },
        { binding: 5, resource: { buffer: uniformBuf } },
      ],
    });

    // 4. Encode & Dispatch Compute Pass
    const commandEncoder = this.device.createCommandEncoder({ label: "CrossAttnEncoder" });
    const passEncoder = commandEncoder.beginComputePass({ label: "CrossAttnPass" });
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, bindGroup);

    const workgroupCount = Math.ceil(numContext / 64);
    passEncoder.dispatchWorkgroups(workgroupCount);
    passEncoder.end();

    // Copy to readback buffers
    commandEncoder.copyBufferToBuffer(outScoresBuf, 0, readbackScoresBuf, 0, numContext * 4);
    commandEncoder.copyBufferToBuffer(outAttnBuf, 0, readbackAttnBuf, 0, numContext * 4);

    this.device.queue.submit([commandEncoder.finish()]);

    // 5. Readback Results from GPU
    await Promise.all([
      readbackScoresBuf.mapAsync(GPUMapMode.READ),
      readbackAttnBuf.mapAsync(GPUMapMode.READ),
    ]);

    const resultScores = new Float32Array(readbackScoresBuf.getMappedRange().slice(0));
    const resultAttn = new Float32Array(readbackAttnBuf.getMappedRange().slice(0));

    readbackScoresBuf.unmap();
    readbackAttnBuf.unmap();

    // Cleanup GPU resources
    contextBuf.destroy();
    queryBuf.destroy();
    baseScoresBuf.destroy();
    outScoresBuf.destroy();
    outAttnBuf.destroy();
    uniformBuf.destroy();
    readbackScoresBuf.destroy();
    readbackAttnBuf.destroy();

    const dt = performance.now() - t0;
    return {
      scores: resultScores,
      attention: resultAttn,
      gpuLatencyMs: parseFloat(dt.toFixed(2)),
    };
  }

  /**
   * Fast CPU fallback when WebGPU adapter is absent.
   */
  cpuFallback(contextTokens, baseScores, queryWords, alpha) {
    const t0 = performance.now();
    const numContext = contextTokens.length;
    const dim = this.embedDim;

    const queryEmbeds = queryWords.map(w => this.embedToken(w, dim));
    const scores = new Float32Array(numContext);
    const attention = new Float32Array(numContext);

    for (let i = 0; i < numContext; i++) {
      const cEmbed = this.embedToken(contextTokens[i], dim);
      let maxSim = -1.0;

      for (let q = 0; q < queryWords.length; q++) {
        const qEmbed = queryEmbeds[q];
        let dot = 0, nC = 0, nQ = 0;
        for (let d = 0; d < dim; d++) {
          dot += cEmbed[d] * qEmbed[d];
          nC += cEmbed[d] * cEmbed[d];
          nQ += qEmbed[d] * qEmbed[d];
        }
        const denom = Math.sqrt(nC) * Math.sqrt(nQ);
        const sim = denom > 1e-6 ? dot / denom : 0;
        if (sim > maxSim) maxSim = sim;
      }

      const qRelevance = Math.max(0.0, Math.min(1.0, (maxSim + 1.0) * 0.5));
      attention[i] = qRelevance;
      scores[i] = alpha * baseScores[i] + (1.0 - alpha) * qRelevance;
    }

    const dt = performance.now() - t0;
    return {
      scores,
      attention,
      gpuLatencyMs: parseFloat(dt.toFixed(2)),
    };
  }
}
