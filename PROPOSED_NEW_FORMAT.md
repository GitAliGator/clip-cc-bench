# Proposed New File Formats for CLIP-CC-Bench

## 1. cross_embedding_model_stats.json → vlm_performance_stats.json

**Current Name Issues:**
- Misleading: Suggests we're comparing embedding models
- Reality: We're evaluating VLM performance using an embedding model as the measurement tool

**New Structure (VLM-focused):**

```json
{
  "vlm_performance": {
    "internvl": {
      "metrics": {
        "coarse_grained": {
          "mean": 0.5695,
          "std": 0.0982,
          "min": 0.0265,
          "max": 0.8433,
          "mean±std": "0.57±0.10"
        },
        "fine_grained": {
          "precision": {
            "mean": 0.3839,
            "std": 0.0618,
            "min": 0.0793,
            "max": 0.5692,
            "mean±std": "0.38±0.06"
          },
          "recall": {
            "mean": 0.3731,
            "std": 0.0726,
            "min": 0.0241,
            "max": 0.5453,
            "mean±std": "0.37±0.07"
          },
          "f1": {
            "mean": 0.3763,
            "std": 0.0631,
            "min": 0.0369,
            "max": 0.5252,
            "mean±std": "0.38±0.06"
          }
        },
        "hybrid": {
          "hm_cf": {
            "mean": 0.4515,
            "std": 0.0726,
            "min": 0.0308,
            "max": 0.6470,
            "mean±std": "0.45±0.07"
          }
        }
      },
      "embedding_model": "nv-embed",
      "num_videos_evaluated": 156,
      "timestamp": "2025-11-19T19:32:02.865044"
    },
    "llava_next_video": { ... },
    "llava_one_vision": { ... }
  },
  "summary": {
    "total_vlms_evaluated": 17,
    "embedding_model": "nv-embed",
    "performance_distribution": {
      "hm_cf": {
        "mean_across_vlms": 0.4640,
        "std_across_vlms": 0.0734,
        "mean±std": "0.46±0.07"
      },
      "coarse_grained": {
        "mean_across_vlms": 0.5693,
        "std_across_vlms": 0.0821,
        "mean±std": "0.57±0.08"
      }
    },
    "timestamp": "2025-11-19T19:45:46.055961"
  }
}
```

**Key Changes:**
- Renamed top-level key from `vlm_name` to `"vlm_performance": { "vlm_name": {...} }`
- Removed misleading `"embedding_model_comparisons"` and `"best_embedding_model"` keys
- Changed `"measurement_tool"` to `"embedding_model"` to clarify it's the measurement instrument
- Added `"mean±std"` field with `avg±std` format for easy reading
- Removed `"ranking"` section (handled by rank_vlms.py instead)
- Changed summary to focus on VLM performance distribution

---

## 2. aggregated_results.csv

**Current Format:**
```csv
Model Name,nv-embed_coarse,nv-embed_fine_f1,nv-embed_hm_cf
llava_one_vision,0.6262,0.4275,0.5063
longvu,0.6329,0.4257,0.5079
```

**New Format (hierarchical headers with avg±std):**

```csv
vlm,nv-embed,nv-embed,nv-embed,nv-embed,nv-embed,gte-qwen2-7b,gte-qwen2-7b,gte-qwen2-7b,gte-qwen2-7b,gte-qwen2-7b
,coarse_grained,fine_grained_precision,fine_grained_recall,fine_grained_f1,harmonic_mean_cf,coarse_grained,fine_grained_precision,fine_grained_recall,fine_grained_f1,harmonic_mean_cf
videollama3,0.68±0.10,0.47±0.08,0.46±0.08,0.47±0.07,0.55±0.08,0.69±0.09,0.48±0.07,0.47±0.07,0.48±0.06,0.56±0.07
llava_one_vision,0.64±0.07,0.48±0.06,0.42±0.06,0.44±0.06,0.52±0.06,0.65±0.05,0.63±0.03,0.61±0.03,0.62±0.03,0.63±0.02
```

**Key Changes:**
- **Row 1**: Embedding model names (repeated for each metric group), lowercase
- **Row 2**: Metric names (coarse_grained, fine_grained_precision, fine_grained_recall, fine_grained_f1, harmonic_mean_cf), lowercase
- **5 metrics per embedding model** for comprehensive analysis
- **No "mean±std" labels** in row 2 - metrics speak for themselves
- **Single file** contains ALL embedding models (columns grow as you add models)
- Combined mean and std into single `avg±std` format
- All lowercase for consistency
- Values rounded to 2 decimal places for readability

---

## 3. detailed_stats.csv → coarse_fine_harmonic_results.csv

**Purpose**: Focused results showing only the 3 key VLM ranking metrics (coarse, fine F1, hybrid).

**Current Format (with separate std columns):**
```csv
,,nv-embed,nv-embed,nv-embed,nv-embed,nv-embed,nv-embed
,VLM,Coarse,Coarse_Std,Fine,Fine_Std,HM,HM_Std
0,llava_one_vision,0.63,0.03,0.43,0.04,0.51,0.03
1,longvu,0.63,0.06,0.43,0.03,0.51,0.03
```

**New Format (hierarchical headers with avg±std):**

```csv
vlm,nv-embed,nv-embed,nv-embed,gte-qwen2-7b,gte-qwen2-7b,gte-qwen2-7b
,coarse_grained,fine_grained_f1,harmonic_mean_cf,coarse_grained,fine_grained_f1,harmonic_mean_cf
videollama3,0.68±0.10,0.47±0.07,0.55±0.08,0.69±0.09,0.48±0.06,0.56±0.07
llava_one_vision,0.64±0.07,0.44±0.06,0.52±0.06,0.65±0.05,0.62±0.03,0.63±0.02
mplug,0.63±0.13,0.46±0.06,0.53±0.08,0.74±0.04,0.65±0.03,0.69±0.02
longvu,0.61±0.10,0.44±0.06,0.51±0.07,0.69±0.03,0.61±0.03,0.65±0.03
```

**Key Changes:**
- **Row 1**: Embedding model names (repeated for each metric group), lowercase
- **Row 2**: Metric names (coarse_grained, fine_grained_f1, harmonic_mean_cf), lowercase
- **Same metrics as aggregated_results.csv** for consistency
- **No "mean±std" labels** in row 2 - metrics speak for themselves
- **Single file** contains ALL embedding models (columns grow as you add models)
- **Removed "Rank" column** - ranking handled by rank_vlms.py
- Combined mean/std into `avg±std` format
- All lowercase for consistency

---

## 4. Summary of Benefits

### For cross_embedding_model_stats.json → vlm_performance_stats.json:
✅ **Clarity:** Focus is on VLM performance, not embedding model comparison
✅ **Easy Reading:** `mean±std` fields show `avg±std` at a glance
✅ **Separation of Concerns:** Rankings handled by rank_vlms.py instead
✅ **Transparency:** Clearly labels embedding model as "embedding_model"

### For CSV files:
✅ **Compact:** Fewer columns, easier to read
✅ **Standard Format:** `avg±std` is common in academic papers
✅ **Clear Headers:** Self-documenting column names
✅ **Paper-Ready:** Can copy-paste directly into LaTeX tables

---

## Next Steps

1. Update `src/utils/result_manager.py` to generate new JSON format
2. Update `src/scripts/rank_vlms.py` to generate new CSV formats
3. Rename file:
   - `cross_embedding_model_stats.json` → `vlm_performance_stats.json`
4. Test with current nv-embed results
5. Verify all 5 embedding models produce consistent format
