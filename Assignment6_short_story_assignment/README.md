# The Cartesian Cut in Agentic AI
### A Review & Reproduction Study

**Course:** [CMPE258 Deep Learning]
**Author:** [Archana Shivashankar]
**Date:** May 2026

---

## 📄 Paper

**Title:** The Cartesian Cut in Agentic AI
**Authors:** Tim Sainburg (Harvard University), Caleb Weinreb (Harvard Medical School)
**Venue:** ICLR 2026
**arXiv:** https://arxiv.org/abs/2604.07745

---

## 🔗 Deliverables

| Deliverable | Link |
|---|---|
| 📰 Medium Article | [https://medium.com/@archana.shivashankar/the-cartesian-cut-in-agentic-ai-b5f49f811734] |
| 🎞️ Slide Deck (SlideShare) | [https://docs.google.com/presentation/d/13H4gHmiUfHRuIcZjXZuPci_LUN8R6E8M_hiOF2B_9p4/edit?usp=sharing] |
| 🎬 Video | [https://drive.google.com/file/d/1pKd3Yd7g3K-V_nl7vOeUhW1idc5Tmkbo/view?usp=sharing] |
| 💻 Reproduction Code |[https://github.com/ashivashankars/CMPE258_Assignments/tree/main/Assignment6_short_story_assignment/autoresearch] |

---

## 📌 Paper Summary

[3–5 sentences in your own words. Example:]

This paper introduces the concept of "Cartesian agency" — the architectural
pattern in which LLM-based agents separate a predictive neural core from an
engineered runtime via a symbolic interface. The authors contrast this with
how brains embed prediction inside layered feedback controllers, arguing that
LLMs invert this relationship by predicting first and retrofitting control
afterward. The paper proposes three agent archetypes — bounded services,
Cartesian agents, and integrated agents — each making different trade-offs
between autonomy, robustness, and oversight. A key finding is that
reinforcement learning on multi-step tool trajectories is gradually eroding
the Cartesian cut as control logic migrates into learned model policies.

---

## 🔬 Reproduction

### What I Reproduced
A demonstration of the three agent archetypes using the autoresearch template.
The goal was to show how control behavior changes depending on where
decision logic lives — in the runtime vs. internalized by the model.

### How to Run
```bash
pip install -r requirements.txt
python reproduction/agent_demo.py
```

### Results
See [reproduction/results/results_summary.md](https://github.com/ashivashankars/CMPE258_Assignments/blob/main/Assignment6_short_story_assignment/autoresearch/autoresearchindexstock/Cartesian_Cut_results_summary.md)

---

## 📊 Key Concepts

| Concept | Description |
|---|---|
| Cartesian Cut | The symbolic boundary between LLM core and external runtime |
| Bounded Service | LLM as stateless function; control fully in runtime |
| Cartesian Agent | ReAct-style; control externalized, passes through text bottleneck |
| Integrated Agent | RL-trained; control logic absorbed into model policy |
| Symbol Bottleneck | All control state must serialize to tokens to cross the cut |

---

## 📁 Repository Structure

```
cartesian-cut-agentic-ai/
│
├── README.md                          ← Most important file
├── article/
│   └── cartesian_cut_article.html    ← Your Medium article (local copy)
├── slides/
│   └── cartesian_cut_slides.pdf      ← Slide deck (upload here too)
├── video/
│   └── presentation.mp4              ← 15–25 min recorded video
├── reproduction/
    ├── README.md                     ← Explains what you reproduced & how to run
    ├── agent_demo.py                 ← Your autoresearch-based code
    └── results/
        └── results_summary.md        ← Your findings/outputs
```
---

## 📚 References

- Sainburg & Weinreb (2026). The Cartesian Cut in Agentic AI. arXiv:2604.07745
- Friston (2005). A theory of cortical responses.
- Gao et al. (2022). PAL: Program-aided Language Models.
- autoresearch template: https://github.com/dlmastery/autoresearch
