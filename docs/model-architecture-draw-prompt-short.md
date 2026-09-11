# Draw Prompt (Short) — Hybrid 3D/2D CNN + Cross-Attention (OCT Glaucoma)

Draw a hybrid CNN–Transformer architecture, left to right. Three inputs: one 3D OCT volume
(1×200×200×200) on top, and two en-face 2D views (slab MIP and full AIP, each 1×224×224) below.
Label every layer and show feature-map sizes shrinking.

**3D CNN branch (top).** Stem: Conv3d(1→32, k3, s1, p1) + GroupNorm + ReLU → 32×200³. Then seven
residual blocks that shrink the maps: identity 32×200³; stride-2 block 64×100³; identity 64×100³;
stride-2 block 128×50³; identity 128×50³; stride-2 block 192×25³; identity 192×25³. After Global
Average Pool 3D and Flatten, the vector is 192-d. Each residual block is pre-activation
1×1 → 3×3 (grouped, groups=8) → 1×1 convolutions with a stride shortcut.

**2D transformer branches (bottom).** Two independent MaxViT-Tiny (ImageNet) towers, one per view.
Each maps 1×224² through a shrinking token grid (224 → 112 → 56 → 28 → 14 → 7) to a 512×7×7 feature
map, then Global Average Pool 2D and Flatten → 512-d vector.

**Projections.** Linear + ReLU: 192→256 for the 3D token (p3D); 512→256 for each 2D token (p2D¹, p2D²).
Stack as three 256-d tokens.

**CrossGate fusion.** Query = p3D (1 token); key/value = LayerNorm([p2D¹; p2D²]) (2 tokens); multi-head
cross-attention with 8 heads and dropout 0.1. Fused output z = p3D + σ(α)·o, where α is a learnable
gate (σ(0.5) = 0.622). The 3D branch is the main path; the 2D branches only modulate it.

**Head.** Fully connected Linear(256→2) → raw logits, then Softmax → class probabilities
(glaucoma / non-glaucoma). Train with class-weighted CrossEntropyLoss.

Label arrows with shapes: (1×200³) → 32×200³ → 64×100³ → 128×50³ → 192×25³ → 192; (1×224²) → 512×7×7
→ 512; tokens 3×256; z 256; logits 2.

Use colors: 3D blue #dbe9f6, 2D green #d9ead3, projections #cfe2f3, CrossGate #f4cccc, head/softmax
#fff2cc, inputs #ffe599. Draw cubes that shrink along the 3D path and grids that shrink along the 2D paths.

Preprocessing: scale the 3D volume by /255; build the two 2D views from the depth axis, resizing to 224².
Optionally add dashed side boxes for X-AI hooks (Grad-CAM 3D/2D, integrated gradients, CrossGate attention).

Total parameters ≈ 58.30 M (3D 0.64 M, 2D 57.08 M, projections 0.31 M, fusion 0.26 M, head 0.0005 M).
Training: AdamW lr 2e-4, wd 1e-4, warmup 5% + cosine, batch 2 + grad-accum 8 (effective 16), bf16 AMP,
30 epochs, patience 6, best validation-AUC checkpoint, temperature scaling, seeds 42/43/44.

Important accuracy notes: the (2,2,1) stride in the code is unused — the real 3D shrink is
200³ → 100³ → 50³ → 25³ (three stride-(2,2,2) stages). The two 2D branches have independent weights.
Token order is [p3D, p2D¹, p2D²]. The model returns raw logits.
