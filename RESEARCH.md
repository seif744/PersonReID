# Person Re-Identification Project — Engineering Handoff (Revised)

## Project Overview

This project is a **CPU-only, fully self-contained multi-camera person re-identification (ReID) system** for recorded CCTV footage.

Its objective is to maintain **stable global identities across multiple fixed cameras** while requiring little or no manual threshold tuning between deployments.

Unlike most academic ReID projects, this system is intended to solve the complete **identity management problem**, not merely appearance matching.

---

# Design Constraints (Non-Negotiable)

The following constraints define every engineering decision.

- CPU-only inference
- Pretrained models only
- No model training
- No fine-tuning
- No cloud inference
- No external identity registration
- Different people every deployment
- Camera positions remain fixed between deployments
- Generalize without recalibrating thresholds for every dataset
- False merges are significantly worse than false splits
- Prefer creating a new identity over incorrectly merging two people

Do **not** recommend:

- GPU-only solutions
- Custom training
- Fine-tuning
- Collecting labeled datasets
- Heavy Vision Transformer-based ReID models unsuitable for CPU deployment

---

# Current Architecture

```
Video
    ↓
YOLO11n
    ↓
ByteTrack
    ↓
TrackEmbedder
    ↓
Appearance Model
    ↓
IdentityService
    ↓
Qdrant
    ↓
Offline Reconciliation
    ↓
Final Render
```

The architecture intentionally separates appearance extraction from identity reasoning.

---

# Architecture Layers

## Appearance Layer

Responsibilities

- Convert one crop into one embedding

Current model

- OSNet x1.0 MSMT17

Output

- 512-dimensional
- L2-normalized embedding

The appearance layer knows nothing about

- identities
- cameras
- gallery
- tracking

Its responsibility ends at feature extraction.

---

## Storage Layer

Current implementation

- Qdrant

Responsibilities

- persistent vector storage
- nearest-neighbour retrieval

Qdrant intentionally remains a dumb vector database.

It should never own identity decisions.

---

## Identity Layer

Owns every identity decision.

Uses

- appearance
- gallery state
- camera
- timing
- track information

This layer is considered the primary area for future engineering improvements.

---

# Current Pipeline

```
Detection
    ↓
Tracking
    ↓
Embedding Extraction
    ↓
Nearest Neighbour Retrieval
    ↓
Camera-aware Re-ranking
    ↓
Verifier
    ↓
Assign Existing Identity
        or
Create New Identity
```

---

# Current Components

## Detection

YOLO11n

---

## Tracking

ByteTrack

---

## Embedding

Current backbone

OSNet x1.0 MSMT17

Previous evaluation

OSNet Market1501

Observed similarity distributions

### MSMT17

Same person

0.70–0.80

Different people

0.48–0.57

### Market1501

Same person

0.65–0.76

Different people

0.70–0.83

MSMT17 provides substantially better separation.

---

## Camera-aware Re-ranking

Already implemented.

Uses

- cosine similarity
- k-reciprocal neighbours
- Jaccard similarity

This should remain part of the system.

---

## Gallery

Current implementation

Persistent Qdrant gallery storing accepted observations.

Current weakness

Nearly every accepted observation eventually contributes to identity representation.

Future improvements should focus on gallery quality rather than gallery size.

---

## Verifier

Current inputs

- cosine similarity
- reranked similarity
- observation count
- recency
- crop quality

Outputs

```
P(same identity)
```

Current implementation is heuristic.

---

## Offline Reconciliation

Already implemented.

Uses

- prototype averaging
- reciprocal nearest neighbours
- union-find
- same-camera overlap exclusion

Produces final global identities.

---

# Engineering Assessment

The architecture itself is considered strong.

The bottleneck is **no longer the embedding network**.

Instead, the remaining weaknesses lie within the **identity decision engine**.

Almost every identity decision still depends primarily on appearance similarity.

Even though the system includes

- reranking
- verifier
- reconciliation

they ultimately rely on the same embedding distribution.

This explains why deployment-specific threshold tuning is still necessary.

---

# Repository Survey

Most academic ReID repositories attempt to improve

> "How similar are these two images?"

This project instead solves

> "Given everything observed over time, do these tracks belong to the same person?"

That distinction should guide all future development.

---

# Repository Assessment

| Repository | Value | Adopt | Notes |
|------------|------:|:----:|------|
| Torchreid | ⭐⭐⭐⭐⭐ | ✅ | Preferred appearance engine. Mature inference, preprocessing, backbone zoo. |
| Layumi Person_ReID_Baseline_Pytorch | ⭐⭐⭐⭐☆ | ✅ Partial | Useful inference implementations and pretrained CNN backbones. Ignore training framework. |
| KPM + Group-Shuffling Random Walk | ⭐⭐☆☆☆ | ⚠ Research only | Interesting local descriptors and graph ideas but limited production benefit. |
| Video Person ReID | ⭐⭐⭐⭐⭐ | ✅ Strongly | Reinforces track-level reasoning and temporal aggregation philosophy. |

---

# Repository-Specific Conclusions

## Torchreid

Adopt

- backbone zoo
- preprocessing
- inference utilities
- feature normalization
- efficient pairwise distance computation

Ignore

- training framework
- losses
- optimizer configuration
- dataset loaders

Torchreid remains the preferred appearance engine.

---

## Layumi Person_ReID_Baseline_Pytorch

Useful primarily as another source of mature CNN backbones and inference implementations.

Potential evaluation candidates

- LightMBN
- OSNet-AIN
- EfficientNet-based ReID
- HRNet

Large Vision Transformers remain low priority because of CPU constraints.

---

## KPM + Group-Shuffling Random Walk

Useful concepts

- local descriptor matching
- graph reasoning

Not recommended for direct integration.

Possible future inspiration

offline graph reconciliation between mature track prototypes.

---

## Video Person ReID

Most valuable repository reviewed.

Not because of its models.

Because of its philosophy.

Important engineering lessons

- tracks are more informative than frames
- temporal aggregation improves robustness
- frame quality matters
- observations should not contribute equally

Its philosophy aligns closely with the future direction of this project.

---

# Proposed Architectural Addition

Introduce a new layer between tracking and identity reasoning.

```
YOLO11n
    ↓
ByteTrack
    ↓
Track Evidence Layer
    ↓
IdentityService
```

---

# Track Evidence Layer

Purpose

Transform noisy frame-level observations into robust track-level evidence before any identity decision is made.

Responsibilities

- accumulate embeddings
- evaluate frame quality
- reject poor observations
- estimate uncertainty
- remove outliers
- build track prototypes
- summarize track statistics

The IdentityService should operate on **track evidence**, not individual embeddings.

---

# Frame Quality Assessment

Every observation should receive a quality score.

Suggested inputs

- bounding-box size
- detector confidence
- sharpness
- motion blur
- visible body percentage
- occlusion
- viewpoint consistency
- embedding consistency
- embedding variance

Low-quality observations should be ignored or heavily down-weighted.

---

# Temporal Aggregation

Current approach

```
Frame

↓

Embedding

↓

Identity
```

Future approach

```
Track

↓

20–30 embeddings

↓

Quality filtering

↓

Outlier removal

↓

Prototype generation

↓

Identity
```

Identity decisions should be made from tracks rather than individual frames whenever possible.

---

# Prototype Construction

Current reconciliation primarily relies on averaged prototypes.

Preferred approach

```
Track

↓

Quality filter

↓

Remove outliers

↓

Cluster embeddings

↓

Representative medoids

↓

Gallery
```

Possible representatives

- frontal
- rear
- left profile
- right profile
- near camera
- far camera

Identity matching should compare against every representative.

Optionally retain a global centroid as an additional coarse descriptor.

---

# Identity Hypothesis Lifecycle

Identity assignment should become a staged process rather than an instantaneous decision.

```
Track Appears

↓

Temporary Identity

↓

Evidence Accumulates

↓

Candidate Matching

↓

Identity Accepted

↓

Gallery Promotion
```

This reduces false merges by allowing additional evidence to accumulate before committing to a global identity.

---

# Evidence-Fusion Identity Engine

Identity reasoning should move away from appearance thresholding.

Instead of

```
Embedding

↓

Cosine Threshold

↓

Identity
```

Move toward

```
Appearance

Color

Relative Height

Camera Topology

Track Consistency

Gallery Confidence

Quality

Decision Margin

↓

Evidence Fusion

↓

Identity Decision
```

Appearance should become one source of evidence rather than the sole decision criterion.

---

# Gallery Management

Current

Accepted observations continuously update identity representations.

Preferred

Only promote observations that

- are sharp
- are sufficiently large
- are minimally occluded
- have stable embeddings
- improve representation diversity

Gallery quality is expected to matter more than gallery size.

---

# Camera Topology

Exploit fixed camera placement.

Rather than only defining impossible transitions,

learn transition-time distributions automatically.

Example

```
Camera A

↓

Camera B

Mean = 12 s

Std = 3 s
```

Use these distributions as evidence during identity verification.

---

# Track Statistics

Every track should expose metadata beyond embeddings.

Examples

- embedding variance
- color variance
- motion smoothness
- normalized height
- observation count
- viewpoint diversity
- track confidence

These statistics provide independent evidence to the IdentityService.

---

# Appearance Backbone Strategy

Current baseline

OSNet x1.0 MSMT17

Highest-priority evaluation

LightMBN

Potential future evaluations

- OSNet-AIN
- EfficientNet-based ReID
- HRNet

Expected benefit from backbone improvements

Moderate.

Improving identity reasoning is expected to provide substantially larger gains than changing CNN backbones.

---

# Research Directions Explicitly Rejected

Avoid significant engineering effort on

- triplet-loss tuning
- ArcFace
- Circle Loss
- Center Loss
- OSM Loss
- attention-loss variants
- dataset preparation
- hyperparameter optimization
- GPU-specific pipelines
- online graph random walks
- large Vision Transformers

These primarily improve appearance embeddings rather than identity reasoning.

---

# Engineering Priorities

### Priority 1

Implement the Track Evidence Layer.

### Priority 2

Deferred identity assignment.

### Priority 3

Frame quality scoring.

### Priority 4

Multi-prototype medoid gallery.

### Priority 5

Evidence-fusion verifier.

### Priority 6

Automatic camera transition modelling.

### Priority 7

Evaluate lightweight CNN backbones

1. LightMBN
2. OSNet-AIN
3. EfficientNet-based ReID
4. HRNet

---

# Design Philosophy

Most academic ReID research focuses on appearance matching.

This project focuses on identity reasoning.

Appearance similarity is an input.

It is **not** identity.

The long-term objective is to evolve from a cosine-threshold-based matching system into a robust identity management engine that reasons over multiple independent sources of evidence while remaining fully CPU-based, pretrained-only, and suitable for production deployment.