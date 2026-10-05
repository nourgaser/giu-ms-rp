"""
screening_decisions.py
======================

Title/abstract screening decisions for the P1 tier, one recorded reason each.

PROVENANCE — read this before quoting any count.
Each of the 111 P1 records was screened at title and abstract against the
inclusion/exclusion criteria in protocol.md, with the reason recorded alongside
the decision. The first pass was drafted with Claude Code and then reviewed;
every row carries its reason precisely so that a call can be checked and
overridden rather than taken on trust.

The set is provisional: screening was at title and abstract only, and full
texts have not been assessed.

Decisions:
  include        -> goes into the review
  include-method -> non-Arabic, kept for the fusion/method lineage (RQ3)
  include-weak   -> defensible but marginal; cite only if space allows
  exclude        -> with reason
"""

DECISIONS: dict[str, tuple[str, str]] = {

    # ---------------------------------------------------------------- 2026
    "GCMA-Net": (
        "include",
        "CRITICAL. Gated cross-modal attention for Arabic MSA on Ar-MuSA "
        "(F1 0.8015). Directly falsifies the draft's 'no cross-modal attention "
        "for Arabic' gap claim. Must be cited and engaged with."),
    "A Directional Semantic Enhancement Approach": (
        "include",
        "CRITICAL. Same group as GCMA-Net; gated fusion + directional semantic "
        "transfer on Ar-MuSA, F1 0.8989 — the current Arabic MSA SOTA and the "
        "calibration target if we run on Ar-MuSA."),
    "A Transformer-Based Approach for Multimodal Arabic": (
        "include",
        "CRITICAL. Tri-modal early fusion (ViT + MarBERT + HuBERT) on Ar-MuSA, "
        "F1 0.7756. Third 2026 Ar-MuSA entry; establishes that the benchmark is "
        "crowded and gives a published early-fusion comparator."),
    "Alignment Challenges in Arabic Multimodal": (
        "include",
        "Arabic MSA alignment — squarely RQ2/RQ3. NOTE: no abstract indexed; "
        "needs manual full-text retrieval before it can be cited properly."),
    "MORAD": (
        "include",
        "CRITICAL as a resource. Moroccan Darija, 2,325 videos, 329 speakers, "
        "SAME 7-emotion taxonomy as EGY-MER and 6x larger. Primary fallback if "
        "EGY-MER access fails. Availability still unverified."),
    "MSA-Moroccan Dialect": (
        "include",
        "Arabic (Darija) multimodal resource, 3,040 podcast samples with aligned "
        "text/audio/visual. Sentiment labels, so a secondary fallback."),
    "Arabic Multimodal Dataset for Aspect Based Sentiment": (
        "include",
        "AraMABSA — Arabic multimodal affect resource, 1,721 text-image pairs "
        "with reported IAA. Text+image only (no audio), so a partial comparator."),
    "Deep Hybrid CNN-LSTM Architecture with Mel-MFCC-Chroma": (
        "include",
        "Egyptian Arabic emotion recognition on the EYASE dataset (4 classes). "
        "Unimodal (speech), but it surfaces EYASE as another Egyptian affective "
        "resource and gives an audio-only comparator for the Egyptian dialect."),
    "CAGMamba": (
        "include-method",
        "2026 alternative to Transformer cross-modal attention (Mamba, linear "
        "rather than quadratic). A direct competitor mechanism for RQ3 and an "
        "obvious 'why not this instead?' question at the meeting."),
    "FMSA: Few-shot Multimodal Sentiment Analysis": (
        "include-method",
        "IEEE TAFFC. Few-shot MSA under data scarcity — our exact regime (363 "
        "samples). Strongest of the 2026 few-shot MSA papers found."),
    "Tri-Subspaces Disentanglement": (
        "include-method",
        "Factorises features into shared/submodal/specific subspaces — a "
        "representational alternative to attention-based fusion, relevant to RQ3."),
    "MulMoSenT": (
        "include-method",
        "Cross-attention fusion for Bengali, another low-resource language. "
        "Useful precedent that cross-attention transfers beyond English. NOTE: "
        "the Mendeley record points to the real paper in Information Fusion — "
        "cite that, not the data record."),
    "ViMACSA-GAT": (
        "include-weak",
        "Vietnamese MSA with cross-modal fusion via graph attention. Parallel "
        "low-resource evidence; cite only if the low-resource argument needs "
        "more than MulMoSenT."),
    "LAPM-RA": (
        "include-weak",
        "Few-shot MSA via reward-adaptive prompt learning. Same territory as "
        "FMSA but weaker venue; redundant unless the few-shot section expands."),
    "AG-MSA": (
        "include-weak",
        "Adaptive gated prompt learning, few-shot MSA. Third paper in the same "
        "few-shot/prompt cluster — keep only to show the trend is broad."),
    "Multilingual multimodal cyberbullying": (
        "include-weak",
        "Not affect per se, but contributes an explicit early/late/hierarchical "
        "fusion taxonomy with uncertainty gating — usable in the fusion-strategy "
        "comparison if that section needs breadth."),

    "BanglaMUSE": (
        "exclude",
        "Non-Arabic dataset (Bangla) with no method contribution. Parallel "
        "low-resource effort but nothing transfers to RQ1-RQ4."),
    "AraImgStory-10K": (
        "exclude",
        "Arabic and multimodal, but the task is story *generation*; emotion "
        "labels are incidental. Not affect recognition."),
    "Arabic Sign Language Recognition": (
        "exclude",
        "Non-affective task (gesture-to-text). Fails inclusion criteria."),
    "Multimodal AI Screening of Developmental Language Disorder": (
        "exclude",
        "Clinical screening for DLD, not affect recognition, despite being "
        "Arabic and multimodal."),
    "Automated MoCA scoring": (
        "exclude",
        "Cognitive (MCI) screening, not affect recognition."),
    "VISH-DARIJA-TTS": (
        "exclude",
        "Vishing / social-engineering detection. Non-affective, and the audio "
        "is synthetic TTS."),
    "Exploring machine learning and computational intelligence for emotion": (
        "exclude",
        "Emotion recognition from *physiological* signals (GSR/PPG) during "
        "Quranic recitation. Modality set disjoint from text/audio/video, so it "
        "does not inform the fusion question."),
}

# ------------------------------------------------------------------ 2025
DECISIONS.update({

    # --- Arabic multimodal resources: the most valuable find in this tier ---
    "EGY-MER": (
        "include",
        "The target dataset. Already in references.bib."),
    "Ar-MuSA": (
        "include",
        "Already in references.bib. Now also the benchmark the three 2026 "
        "Arabic cross-modal-attention papers compete on."),
    "UniTextFusion": (
        "include",
        "Already in references.bib. Generative early-fusion + LoRA baseline."),
    "Amd'SaEr": (
        "include",
        "MAJOR FIND. ACM TALLIP — an Arabic multimodal dataset covering BOTH "
        "sentiment and emotion recognition. Strong venue, directly on RQ1, and "
        "absent from the current bibliography. Another EGY-MER alternative."),
    "MDER-MA": (
        "include",
        "MAJOR FIND. Multimodal emotion dataset for Moroccan Arabic (Data in "
        "Brief). A second Moroccan emotion resource alongside MORAD — widens "
        "the fallback options considerably."),
    "Multi-Modal Spoken-Arabic Dataset Collection for Emotion": (
        "include",
        "Another Arabic multimodal emotion resource with explicit attention to "
        "dialectal/cultural variation. Directly RQ1."),
    "AraSER-Fusion": (
        "include",
        "Multimodal Arabic SER with audio-text joint embeddings (Arabian J. Sci. "
        "Eng.). NOTE: no abstract indexed — needs manual retrieval."),
    "Arabic Multimodal Machine Learning: Datasets, Applications": (
        "include",
        "ESSENTIAL. A dedicated survey of Arabic multimodal ML — datasets, "
        "approaches, challenges. The natural anchor for the Arabic-landscape "
        "section and a check on our own coverage."),
    "Multimodal Arabic Sarcasm Detection Using CNN and BiLSTM": (
        "include",
        "Directly RQ4: multimodal Arabic sarcasm on Ar-MuSA, fusing text, audio "
        "and image. The only multimodal Arabic sarcasm work found — central to "
        "the incongruence argument."),
    "Deciphering Emotions in Children": (
        "include-weak",
        "Multimodal LLMs for emotion recognition in Arabic educational content. "
        "Evaluation rather than method, but evidence on Arabic multimodal "
        "affect with LLMs."),
    "Sentiment Analysis for Low-Resource and Arabic Dialects": (
        "include",
        "Review of SA across low-resource languages and Arabic dialects — "
        "useful framing for the dialectal-challenge section."),
    "EmotionNet": (
        "include-weak",
        "Arabic SER (unimodal audio, KEDAS dataset). Useful only as a unimodal "
        "Arabic comparator; surfaces KEDAS as another Arabic affective resource."),
    "Multimodal Emotion-Aware ASR": (
        "include",
        "Already in references.bib (Dabbabi2025). Low-rank tensor fusion."),

    # --- fusion / method lineage (non-Arabic), RQ3 ---
    "RAFT": (
        "include-method",
        "Most-cited 2025 paper in the pool (84). Robust adversarial fusion "
        "transformer addressing modality noise and missing channels — directly "
        "the fusion-mechanism lineage."),
    "DLF: Disentangled-Language-Focused": (
        "include-method",
        "AAAI 2025, 60 citations. Language-focused disentanglement — the closest "
        "high-profile analogue to our text-as-query design, and the strongest "
        "'why not this instead?' comparator."),
    "MSE-Adapter": (
        "include-method",
        "AAAI 2025. Lightweight adapter giving LLMs MSA/ERC capability — "
        "parameter-efficient fusion, the same motivation as our LoRA argument."),
    "Cross-lingual Multimodal Sentiment Analysis for Low-Resource": (
        "include-method",
        "HIGH VALUE. Cross-lingual MSA for low-resource languages via language-"
        "family disentanglement. Closest published framing to our transfer "
        "question."),
    "Hierarchical Text-Guided Refinement Network": (
        "include-method",
        "Text-guided refinement of non-text features — mechanically very close "
        "to Tri-CAN's text-as-query design. Must be cited and distinguished."),
    "Leveraging pre-trained self-supervised learning models for low-resource": (
        "include-method",
        "Frozen self-supervised encoders for low-resource multimodal text+speech "
        "— precedent for exactly the frozen-encoder protocol we adopted."),
    "Adversarial Multimodal Sentiment Analysis with Cross-Modal and Cross-Domain": (
        "include-method",
        "Attention-based fusion under data scarcity and domain shift — our "
        "small-data regime."),
    "MGMR-Net": (
        "include-weak",
        "Mamba-guided fusion with text-centric design. Overlaps CAGMamba (2026); "
        "keep one of the two unless the Mamba alternative gets its own paragraph."),
    "CLARA: Enhancing Multimodal Sentiment Analysis": (
        "include-weak",
        "Co-attention, parameter-efficient vision-language fusion. NOTE: appears "
        "twice in the pool (paper + Zenodo software record) — cite once."),
    "Quantum Infused Multi-Modal Sentiment Analysis": (
        "include-weak",
        "Urdu (low-resource) multimodal via cross-modal transformer. The quantum "
        "framing is irrelevant to us; the low-resource cross-modal evidence is not."),
    "Uncertainty-Aware Collaborative System of Large and Small Models": (
        "include-weak",
        "Efficiency-motivated large+small collaboration for MSA. Cite only if "
        "the compute-constraint discussion needs support."),
    "Multi-Modality Collaborative Learning for Sentiment Analysis": (
        "include-weak",
        "Cross-modal interaction modelling; a generic comparator in a crowded field."),
    "Improving Multimodal Sentiment Analysis via Modality Optimization": (
        "include-weak",
        "Dynamic primary-modality selection — relevant to our assumption that "
        "text should always be the query. A genuine challenge to that choice."),

    # --- excluded ---
    "phuonglamgithub/CLARA": (
        "exclude", "Duplicate of the CLARA paper (Zenodo software release record)."),
    "Preface: Crossing The Nexus": (
        "exclude", "Journal preface, not a research paper."),
    "MARSAD": (
        "exclude", "Social-media monitoring platform; text-based tooling, not MSA method."),
    "Comprehensive Analysis of Bangla Sarcastic Comments": (
        "exclude", "Text-only sarcasm in Bangla. Neither Arabic nor multimodal."),
    "SmolLab_SEU at MAHED": (
        "exclude", "Arabic hope/hate detection but the authors' subtasks are text-only."),
    "Integrating Deep Learning and Swarm Intelligence": (
        "exclude", "Speech-recognition/optimisation review; off-topic for RQ1-RQ4."),
    "Towards Emotional Image Captioning": (
        "exclude", "Emotional text *generation*, not affect recognition."),
    "Sentiment-Annotated Hibapress": (
        "exclude", "Moroccan Arabic news sentiment, but text-only (headlines)."),
    "Opened Mind": (
        "exclude", "Deployed empathetic-AI mental-health application, not an MSA method."),
    "A Multimodal Bangla Text–Audio Dataset": (
        "exclude", "Non-Arabic dataset with no method contribution."),
    "Multimodal Conversational Agent for Mental Health Support": (
        "exclude", "Application system paper; no comparable evaluation on an MSA benchmark."),
    "ARTIFICIAL INTELLIGENCE IN LITERARY ANALYSIS": (
        "exclude", "Emotion in poetry, text-only, unrelated to multimodal fusion."),
    "Integrating Sarcasm-Rich Arabic Street Slang": (
        "exclude", "Arabic sarcasm but text-only (GPT prompting). Relevant to RQ4 "
                   "framing only; cite in passing if the sarcasm section needs it."),
    "Advancing Multimodal Sentiment Analysis: Models, Datasets": (
        "exclude", "Generic MSA overview, no venue indexed, superseded by the "
                   "Arabic-specific survey we are including."),
    "Exploring Cultural Nuances in Emotion Perception Across 15 African": (
        "exclude", "Cross-linguistic emotion perception in African languages; "
                   "text-based and outside the Arabic scope."),
    "Senti-iFusion": (
        "exclude", "Missing-modality robustness — a different problem from ours; "
                   "the pool already has better-matched fusion comparators."),
    "Structures Meet Semantics": (
        "exclude", "Graph contrastive fusion; redundant with stronger comparators."),
    "DashFusion": (
        "exclude", "Bottleneck fusion; redundant with RAFT/DLF for our purposes."),
    "Hybrid CNN-Mamba Enhancement Network": (
        "exclude", "Missing-modality Mamba variant; redundant with CAGMamba."),
    "DAST-GAN": (
        "exclude", "Adversarial spatio-temporal fusion for unaligned/missing "
                   "modalities; redundant with RAFT."),
})

# -------------------------------------------------------------- pre-2025
DECISIONS.update({

    # --- Arabic multimodal affect: the historical spine of the review ---
    "Multimodal Arabic emotion recognition using deep learning": (
        "include",
        "IMPORTANT. Speech Communication (2023) — Arabic multimodal emotion "
        "recognition in a strong speech venue, absent from the current "
        "bibliography. NOTE: no abstract indexed; retrieve full text."),
    "A multi-modal deep learning system for Arabic emotion recognition": (
        "include",
        "Int. J. Speech Technology (2022). Arabic multimodal emotion system — "
        "direct predecessor to EGY-MER's task. NOTE: no abstract indexed."),
    "A Novel Deep Learning Multi-Modal Sentiment Analysis Model for English and Egyptian": (
        "include",
        "HIGH VALUE. ATFusion — audio+text multimodal sentiment for **Egyptian "
        "Arabic** specifically. The closest prior work to our target dialect and "
        "modality set outside EGY-MER itself."),
    "Multimodal Sentiment Analysis of Arabic Videos": (
        "include",
        "2018, 16 citations. Early Arabic multimodal video sentiment — needed "
        "for the historical arc before the 2025 watershed."),
    "Toward’s Arabic Multi-modal Sentiment Analysis": (
        "include",
        "2018, 16 citations. Foundational Arabic MSA; PDF already in rp/refs/."),
    "Towards Arabic Multimodal Dataset for Sentiment Analysis": (
        "include",
        "Already in references.bib (Haouhat2023)."),
    "Arabic language investigation in the context of unimodal and multimodal": (
        "include",
        "Already in references.bib (Youcef2021)."),
    "Multimodal Arabic Sentiment Analysis for Video Opinion Mining": (
        "include-weak",
        "2019 Arabic multimodal video opinion mining. Institutional repository, "
        "low visibility, but fills the pre-2020 gap in the Arabic timeline."),
    "Arabic Multimodal Emotion Recognition Using Deep Learning": (
        "include-weak",
        "MSc thesis (American University of Sharjah, 2022). Grey literature — "
        "cite only if a peer-reviewed version cannot be found."),
    "Speech Emotion Recognition in Multimodal Environments with Transformer": (
        "include-weak",
        "Transformer SER on Arabic and English audio datasets. Unimodal-audio "
        "comparator for the Arabic side."),
    "Automatic sarcasm detection in Arabic tweets": (
        "include-weak",
        "Text-only Arabic sarcasm, but a useful resources-and-approaches survey "
        "for the RQ4 framing where the current draft cites only one source."),

    # --- fusion / cross-modal attention lineage (RQ3) ---
    "Efficient Multimodal Transformer With Dual-Level Feature Restoration": (
        "include-method",
        "MOST-CITED IN THE POOL (222). IEEE TAFFC. Efficient multimodal "
        "transformer — the strongest single comparator for a cross-modal "
        "attention architecture."),
    "Cross-Modal Enhancement Network": (
        "include-method",
        "IEEE TMM, 142 citations. Core cross-modal interaction lineage between "
        "MulT (2019) and the 2025-26 work."),
    "Cross-Modality Gated Attention Fusion": (
        "include-method",
        "KEY. Gated attention fusion for MSA (2022) — the direct methodological "
        "ancestor of GCMA-Net. Shows the Arabic 2026 work is applying an "
        "established mechanism, which sharpens what is and isn't novel."),
    "Text-oriented Modality Reinforcement Network": (
        "include-method",
        "Text-oriented reinforcement over unaligned sequences — the same "
        "text-as-anchor premise as Tri-CAN. Must be cited and distinguished."),
    "Diversity and Balance": (
        "include-method",
        "IEEE TAFFC 2024. Multimodal-prefixed + cross-modal attention; addresses "
        "modality imbalance, which is our problem on an imbalanced corpus."),
    "SCANET": (
        "include-method",
        "Sparse and cross-attention fusion, explicitly motivated by text's "
        "higher-order semantics dominating other modalities — our design premise."),
    "Tri-CLT": (
        "include-method",
        "Tri-modal contrastive + transformer representations. Name and structure "
        "are close enough to Tri-CAN that we must distinguish ours explicitly."),
    "Syntax-aware Hybrid prompt model for Few-shot": (
        "include-method",
        "Few-shot MSA under small labelled data — our regime."),
    "A Unified Self-Distillation Framework": (
        "include-method",
        "AAAI 2024, 51 citations. Self-distillation for MSA; a strong "
        "data-efficiency comparator."),
    "Cross-Modality Dual-Attention Fusion Network": (
        "include-weak",
        "CMDAF — dual-attention fusion; a comparator in a crowded family."),
    "Multimodal Sentiment Analysis in Realistic Environments Based on Cross-Modal Hierarchical": (
        "include-weak",
        "Hierarchical cross-modal fusion under noise; cite if the fusion "
        "taxonomy section needs another rung."),
    "Multimodal Mutual Attention-Based Sentiment Analysis Framework": (
        "include-weak",
        "Mutual (bidirectional) attention — relevant to the ablation on "
        "attention direction, since Tri-CAN is text-directed only."),
    "Which is Making the Contribution": (
        "include-weak",
        "Modulating unimodal vs cross-modal dynamics — supports the ablation "
        "asking how much the cross-modal path actually contributes."),
    "A multimodal approach to cross-lingual sentiment analysis with ensemble": (
        "include-weak",
        "Scientific Reports, 180 citations. Cross-lingual multimodal SA; high "
        "visibility, loosely coupled to our question."),
    "Enhanced Video Analytics for Sentiment Analysis Based on Fusing Textual": (
        "include-weak",
        "IEEE Access 2020, tri-modal fusion for video sentiment. Verify whether "
        "the data is Arabic before citing it as such."),

    # --- excluded ---
    "Bienvenidos a la conversación": (
        "exclude", "Science-education translanguaging commentary. Retrieved by a "
                   "false keyword match; entirely off-topic."),
    "Editorial: Emerging applications of text analytics": (
        "exclude", "Journal editorial, not a research paper."),
    "=Multimodal Sentiment Analysis": (
        "exclude", "Generic anthology chapter with no method contribution."),
    "Discourse of Socio-political Crisis and the Pandemic": (
        "exclude", "Linguistic discourse analysis of Arabic cartoons; not "
                   "computational affect recognition."),
    "Translating irony into Arabic": (
        "exclude", "Translation-studies analysis of dubbing. Interesting on "
                   "Egyptian irony but not a computational contribution."),
    "Sentiment Analysis in Multiple Languages": (
        "exclude", "Generic multilingual SA review; superseded by the Arabic-"
                   "specific surveys we include."),
    "Video Demographic Analytics of Social Media Users": (
        "exclude", "Demographic inference, not affect recognition."),
    "Empirical Comparison between Deep and Classical Classifiers for Speaker": (
        "exclude", "Speaker verification, not emotion recognition."),
    "DUCS at SemEval-2022": (
        "exclude", "Text-only sarcasm shared-task entry."),
    "Cultural-Aware AI Model for Emotion Recognition": (
        "exclude", "No venue indexed and no benchmark comparison; the cultural "
                   "argument is better served by the Arabic-specific sources."),
    "Mixture-of-Prompt-Experts": (
        "exclude", "Multimodal semantic understanding, not affect."),
    "PowMix": (
        "exclude", "Regularisation technique; orthogonal to the fusion question."),
    "Self-HCL": (
        "exclude", "Contrastive multitask MSA; redundant with stronger comparators."),
    "Learning Modality Consistency and Difference Information": (
        "exclude", "Multitask consistency/difference learning; redundant."),
    "Improving the Modality Representation with Multi-View Contrastive": (
        "exclude", "Contrastive representation learning; redundant."),
    "Novel Dual-Channel Long Short-Term Memory Compressed Capsule": (
        "exclude", "Unimodal speech emotion with capsule networks; no fusion "
                   "contribution and not Arabic-specific."),
})
