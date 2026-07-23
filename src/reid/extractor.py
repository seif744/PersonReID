"""
extractor.py  --  STAGE 5:  ReID feature extraction.

============================ THE BIG PICTURE ================================
This module has EXACTLY ONE job:

    OpenCV BGR person crop  ->  L2-normalized 512-d embedding.

That's it. It does NOT assign identities, search a database, or know anything
about cameras, tracks, timestamps, YOLO, ByteTrack, or Qdrant. It is a pure
function wrapped in a class (the class only exists so we load the heavy model
weights ONCE and reuse them for every crop).

WHY the strict boundary: an embedding is a reusable measurement of "what this
person looks like." Deciding WHO they are (matching, gallery, global ids) is a
separate, stateful concern with its own failure modes (camera topology, time
windows, motion). Mixing the two is how ReID systems rot. Keep this dumb.

--------------------------------- THE MODEL --------------------------------
OSNet-AIN x1_0 (torchreid). Same OSNet family as before, with Adaptive Instance
Normalization layers added for better generalization to unseen (out-of-domain)
camera footage -- swapped in from plain osnet_x1_0 for exactly that reason. We
depend ONLY on torchreid's model DEFINITION, not on its training machinery --
so this file stays a thin inference path we can later re-point at a vendored
osnet.py for TensorRT without changing the public API.

Embedding dimension is 512. This is a property of the CHECKPOINT (osnet_ain_x1_0's
feature layer width, same as osnet_x1_0), not a free choice. It is also the
vector size the Qdrant collection will use downstream, so if the backbone ever
changes to a different width, that number propagates outward.

------------------------------ PREPROCESSING -------------------------------
The model was trained on inputs prepared a specific way. Inference MUST match
training preprocessing byte-for-byte in spirit, or embeddings silently degrade
(no error -- just worse matching). The required recipe, in order:

  1. BGR -> RGB        OpenCV gives BGR; the model was trained on RGB.
  2. resize to 256x128 (H x W). OSNet's canonical person-ReID input. People are
                       tall and narrow, hence the 2:1 aspect ratio. Feeding a
                       different size still runs (OSNet is fully convolutional +
                       global pooling) but shifts the feature statistics away
                       from training -> worse accuracy. Pin it.
  3. /255              scale 0..255 -> 0..1.
  4. (x-mean)/std      per-channel ImageNet standardization. These constants are
                       the ImageNet TRAIN-SET channel statistics. ReID training
                       started from ImageNet weights and never changed the
                       normalization, so we must reuse the exact same numbers.
                       Change them and every activation drifts off-distribution.

NOTE we keep ImageNet *normalization* but NOT ImageNet *weights* -- the loaded
checkpoint is fine-tuned for person ReID, which is what makes the embedding
separate identities instead of object categories.

------------------------------ WHY L2-NORMALIZE ----------------------------
We divide each embedding by its L2 norm so every vector lands on the unit
hypersphere. Then cosine similarity between two people is just a dot product,
and Euclidean distance becomes a monotonic function of cosine -- so downstream
matching (and Qdrant's COSINE metric) is well-defined and scale-invariant. The
raw magnitude of an OSNet feature carries no identity information; only its
DIRECTION does.
============================================================================
"""

from typing import List, Optional

import cv2
import numpy as np
import torch
import torchreid


# --- Constants (see module docstring for the "why" of each) ----------------

# Model input as (Height, Width). OSNet's canonical person-ReID resolution.
INPUT_SIZE = (256, 128)

# Embedding width produced by osnet_ain_x1_0's feature layer -- same 512 width
# as osnet_x1_0, verified against the checkpoint's classifier.weight shape.
EMBEDDING_DIM = 512

# ImageNet train-set per-channel statistics, RGB order, on the 0..1 scale.
# Shaped (3,1,1) so they broadcast over a (3,H,W) tensor.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class ReIDExtractor:
    """
    Loads OSNet-AIN x1_0 once and turns person crops into embeddings.

    Typical use:
        extractor = ReIDExtractor(weights="weights/osnet_ain_x1_0.pth")
        emb = extractor.extract(person_crop)     # (512,) float32, ||emb|| == 1
    """

    def __init__(self, weights: str, device: Optional[str] = None):
        """
        weights : path to a person-ReID OSNet checkpoint (.pth). NOT ImageNet
                  weights -- those do not separate identities.
        device  : "cuda", "cpu", or None to auto-pick. Kept explicit so the
                  same code runs on the CPU dev box and a GPU deployment box
                  without edits.
        """
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Build the ARCHITECTURE only (pretrained=False): we supply our own
        # ReID-trained weights below. num_classes here is irrelevant -- it only
        # sizes the classifier head, which we deliberately discard.
        model = torchreid.models.build_model(
            name="osnet_ain_x1_0",
            num_classes=1000,
            loss="softmax",
            pretrained=False,
        )

        # Load our checkpoint EXPLICITLY so the load is transparent and owned by
        # this file (rather than hidden inside a framework helper). The only two
        # keys we DROP are classifier.{weight,bias}: they are the training-time
        # identity head, meaningless for features.
        # weights_only=False: PyTorch >=2.6 defaults to weights_only=True, which
        # rejects the numpy scalars pickled into these legacy torchreid
        # checkpoints. Safe here -- we only ever point this at checkpoints we
        # downloaded ourselves from the trusted torchreid Model Zoo.
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        # Some Model Zoo checkpoints are the bare state_dict; others (like this
        # AIN one) are a full training checkpoint wrapping it under "state_dict",
        # with a "module." prefix left over from DataParallel training.
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model_sd = model.state_dict()
        to_load = {
            k: v for k, v in state_dict.items()
            if k in model_sd and model_sd[k].shape == v.shape
        }
        dropped = [k for k in state_dict if k not in to_load]
        if dropped != ["classifier.weight", "classifier.bias"]:
            # Anything else missing means the checkpoint doesn't match osnet_ain_x1_0
            # -- fail loud rather than emit garbage embeddings.
            raise RuntimeError(
                f"Unexpected checkpoint keys dropped: {dropped}. "
                f"Expected only the classifier head. Wrong architecture or file?"
            )
        model.load_state_dict(to_load, strict=False)

        # eval() is REQUIRED for two reasons:
        #   1. it freezes BatchNorm to use running stats (not per-batch stats),
        #      so one crop and a batch of crops give identical embeddings;
        #   2. in eval mode OSNet's forward returns the 512-d FEATURE, whereas
        #      in train mode it returns classifier logits.
        model.eval()
        self.model = model.to(self.device)

    @torch.no_grad()
    def extract(self, crop: np.ndarray) -> np.ndarray:
        """
        One BGR crop -> (512,) L2-normalized float32 embedding.

        Convenience wrapper over extract_batch for the common single-crop call.
        """
        return self.extract_batch([crop])[0]

    @torch.no_grad()
    def extract_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        A list of BGR crops -> (N, 512) L2-normalized float32 embeddings.

        Batching is a REAL production need, not gratuitous abstraction: a single
        frame yields many person boxes, and running them through the network in
        one forward pass (instead of one call each) is a large GPU throughput
        win with hundreds of cameras. On CPU it still avoids Python-loop
        overhead. So the batch path is primary; extract() just calls it with N=1.

        @torch.no_grad() disables autograd -- we never train here, so we skip
        building the graph: less memory, faster.
        """
        if len(crops) == 0:
            # Explicit empty case: return a well-shaped (0, 512) array so callers
            # can concatenate results without special-casing.
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        batch = torch.stack([self._preprocess(c) for c in crops])
        batch = batch.to(self.device)

        features = self.model(batch)                      # (N, 512), raw features

        # L2-normalize along the feature dim. eps guards the degenerate case of
        # an all-zero feature (never seen in practice) from producing NaNs.
        features = torch.nn.functional.normalize(features, p=2, dim=1, eps=1e-12)

        # Back to CPU/NumPy for the rest of the (non-torch) pipeline.
        return features.cpu().numpy().astype(np.float32)

    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        """
        One BGR uint8 crop -> normalized (3, 256, 128) float32 tensor on CPU.

        Kept as a private method (not a standalone util) because it is only ever
        meaningful paired with this exact model's expectations.
        """
        if crop is None or crop.size == 0:
            # A zero-area box (can happen at frame edges) has no pixels to
            # describe. Fail loud -- silently embedding a blank crop would inject
            # a meaningless vector into matching.
            raise ValueError("Empty crop passed to ReIDExtractor.")

        # 1. BGR -> RGB.
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # 2. Resize to (W, H) -- cv2 takes (width, height), our constant is
        #    (H, W), hence the reversed indexing. INTER_LINEAR matches the
        #    bilinear interpolation used during training.
        rgb = cv2.resize(rgb, (INPUT_SIZE[1], INPUT_SIZE[0]),
                         interpolation=cv2.INTER_LINEAR)

        # 3. HWC uint8 -> CHW float32 in 0..1.
        tensor = torch.from_numpy(rgb).float().div_(255.0).permute(2, 0, 1)

        # 4. Per-channel ImageNet standardization.
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        return tensor
