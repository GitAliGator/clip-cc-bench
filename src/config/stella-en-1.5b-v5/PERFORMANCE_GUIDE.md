# Stella-en-1.5b-v5 Performance Guide

## Overview

Stella-en-1.5b-v5 is a lightweight, high-performance embedding model that provides excellent text similarity computation with lower memory requirements compared to BGE-ICL. This guide provides comprehensive recommendations for optimal configuration and performance.

## Model Specifications

- **Model Name**: `dunzhang/stella_en_1.5B_v5`
- **Parameters**: 1.5 billion
- **Architecture**: Based on Alibaba-NLP/gte-large-en-v1.5 and gte-Qwen2-1.5B-instruct
- **Embedding Dimensions**: 512, 768, 1024, 2048, 4096, 6144, 8192 (default: 1024)
- **Max Sequence Length**: 8192 tokens (recommended: 512)
- **Model Type**: Sentence Transformer with MRL (Multiple Representation Learning)

## Optimal Configuration Settings

### Core Parameters

```yaml
encoder:
  batch_size: 16                    # Recommended starting point
  max_length: 512                   # Optimal for performance
  embedding_dimension: 1024         # Best performance/efficiency balance
  normalize_embeddings: true        # Always recommended
  prompt_type: "s2s"               # For similarity tasks
```

### Device and Memory Settings

```yaml
processing:
  device: "cuda:0"                  # Use GPU when available
  clear_cache_interval: 25          # Clear cache every 25 iterations
  force_gc_interval: 50             # Force garbage collection
```

## Performance Recommendations

### Batch Size Optimization

| GPU Memory | Recommended Batch Size | Expected VRAM Usage |
|------------|----------------------|-------------------|
| 8GB        | 8-12                 | ~4-5GB            |
| 12GB       | 16-20                | ~6-7GB            |
| 16GB+      | 24-32                | ~8-10GB           |
| 24GB+      | 32-48                | ~12-15GB          |

### Text Length Guidelines

- **Optimal**: 512 tokens or less
- **Good**: 512-1024 tokens
- **Acceptable**: 1024-2048 tokens
- **Not Recommended**: >2048 tokens (performance degradation)

### Embedding Dimension Trade-offs

| Dimension | Performance | Memory Usage | Use Case |
|-----------|-------------|--------------|----------|
| 512       | Good        | Lowest       | Resource-constrained environments |
| 768       | Better      | Low          | Balanced efficiency |
| **1024**  | **Excellent** | **Medium**   | **Recommended default** |
| 2048      | Excellent   | High         | High-precision requirements |
| 4096+     | Marginal gain | Very High  | Research/experimental |

**Note**: 1024d provides 99.9% of the performance of 8192d with significantly lower memory usage.

## Prompts and Task Configuration

### Prompt Types

1. **Sentence-to-Sentence (s2s)**: `"Instruct: Retrieve semantically similar text.\nQuery: "`
   - Use for: Text similarity, paraphrase detection, semantic matching
   - Recommended for most similarity evaluation tasks

2. **Sentence-to-Passage (s2p)**: `"Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "`
   - Use for: Information retrieval, question-answering, passage ranking

### Example Configuration for Different Tasks

```python
# For video summary similarity (recommended)
config = {
    'prompt_type': 's2s',
    'max_length': 512,
    'normalize_embeddings': True
}

# For document retrieval
config = {
    'prompt_type': 's2p',
    'max_length': 1024,
    'normalize_embeddings': True
}
```

## Memory Management

### GPU Memory Optimization

```python
# Clear cache regularly
if iteration % 25 == 0:
    torch.cuda.empty_cache()
    gc.collect()

# Monitor memory usage
def get_memory_usage():
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,
            'reserved': torch.cuda.memory_reserved() / 1024**3
        }
```

### Caching Strategy

```yaml
embedding:
  cache_embeddings: true
  cache_ground_truth_embeddings: true
  cache_dir: "results/encoders/cache"
```

Benefits:
- Avoids re-computing ground truth embeddings
- Significantly faster for repeated evaluations
- Disk space usage: ~1-2GB for typical datasets

## Performance Benchmarks

### Expected Processing Times (GPU)

| Operation | Batch Size 16 | Batch Size 32 | Notes |
|-----------|---------------|---------------|-------|
| Text encoding (512 tokens) | ~0.1s/batch | ~0.15s/batch | Including prompt processing |
| Similarity computation | ~0.01s/pair | ~0.005s/pair | Vectorized operations |
| Memory allocation | ~4GB | ~6GB | Peak usage during encoding |

### Comparison with BGE-ICL

| Metric | Stella-en-1.5b-v5 | BGE-ICL | Advantage |
|--------|-------------------|---------|-----------|
| Memory Usage | ~4GB | ~8GB | 50% reduction |
| Inference Speed | Fast | Medium | ~30% faster |
| Model Size | 3GB | 5GB | Smaller footprint |
| Performance | Competitive | Excellent | Minimal difference |

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch_size to 8 or 12
   - Reduce max_length to 256
   - Clear cache more frequently

2. **Slow Performance**
   - Ensure GPU usage: check `torch.cuda.is_available()`
   - Increase batch_size if memory allows
   - Enable mixed precision training

3. **Poor Similarity Scores**
   - Verify prompt_type matches your task
   - Ensure normalize_embeddings=true
   - Check text preprocessing

### Optimization Checklist

- [ ] GPU acceleration enabled
- [ ] Optimal batch size configured
- [ ] Appropriate text length limits
- [ ] Caching enabled for ground truth
- [ ] Regular memory cleanup
- [ ] Proper prompt type selected
- [ ] Embedding normalization enabled

## Advanced Configuration

### Custom Embedding Dimensions

To use non-1024 dimensions, modify the model's `modules.json`:

```bash
# Download model locally first
git clone https://huggingface.co/dunzhang/stella_en_1.5B_v5

# Edit modules.json
# Replace "2_Dense_1024" with "2_Dense_512" (or desired dimension)
```

### Production Deployment

```python
class ProductionStellaEncoder:
    def __init__(self):
        self.model = SentenceTransformer(
            "dunzhang/stella_en_1.5B_v5",
            device="cuda",
            trust_remote_code=True
        )

    def encode_batch(self, texts, batch_size=16):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
```

## Monitoring and Metrics

### Key Performance Indicators

```python
# Track these metrics
metrics = {
    'avg_encoding_time': time_per_batch,
    'memory_efficiency': peak_memory / batch_size,
    'throughput': texts_per_second,
    'similarity_accuracy': avg_similarity_score
}
```

### Health Checks

```python
# Regular health checks
def health_check():
    # Memory usage
    memory_ok = torch.cuda.memory_allocated() < 0.8 * torch.cuda.max_memory_allocated()

    # Model responsiveness
    test_embedding = model.encode(["test text"])

    return memory_ok and test_embedding is not None
```

## Conclusion

Stella-en-1.5b-v5 offers excellent performance for text similarity tasks with lower resource requirements than larger models like BGE-ICL. Following these recommendations will ensure optimal performance in production environments while maintaining competitive accuracy.