Contributing to ACI (Always-On Consciousness-Inspired AI)
=========================================================

First off, thank you for your interest in contributing!

This project explores a neuro-inspired, memory-centric architecture for an always-on, consciousness-inspired agent. There is a rough working implementation of the DMN-style loop already running in a Jupyter Notebook. The next major milestone is a reimplementation targeting:

-   Isaac Sim for grounded sensory experience and world interaction

-   Google Colab for accessible, reproducible experiments

Until that reimplementation lands, contributions that refine algorithms, interfaces, and evaluation strategies are very welcome.

Project Status
--------------

-   Current: A prototype DMN loop with module stubs and basic orchestration in a notebook.

-   In progress: Designing the memory system and grounding interfaces for Isaac Sim.

-   Near-term goals:

    -   Define concrete data schemas for memory nodes/edges.

    -   Specify z_self update equations and calibration/safety metrics.

    -   Implement HC expansion controls and compute budgets.

    -   Provide Colab and Isaac Sim scaffolding.

How to Contribute
-----------------

-   License: MIT

-   Workflow: Fork → Branch → Commit → PR

-   Requirements:

    -   Explain the purpose of your change clearly in the PR description.

    -   Include a brief rationale: what problem it solves, how it affects the system, any trade-offs.

    -   If you modify algorithms, add notes on assumptions, expected complexity, and defaults.

    -   If you introduce new parameters, document sensible defaults and bounds.

Good First Contribution Ideas
-----------------------------

-   Algorithm refinement

    -   Clarify and/or implement scoring features (coherence, novelty, epistemic gain).

    -   Propose/implement neuromodulator gating policies (e.g., beam width schedules).

    -   Draft consolidation thresholds and causal edge confidence calculations.

-   Memory schemas

    -   Propose a minimal node/edge schema for episodic → semantic → autobiographical layers.

    -   Sketch a hybrid storage approach (vector + graph + columnar attributes).

-   Safety and calibration

    -   Define a simple calibration loop (e.g., reliability diagrams, Brier/NLL updates).

    -   Integrate safety penalties into decoding/selection with transparent logging.

-   Evaluation

    -   Add ablation plans and metrics for identity coherence, abstraction emergence, and safety.

    -   Provide small synthetic tasks for pipeline health checks.

Pull Request Guidelines
-----------------------

-   Keep PRs focused and reviewable.

-   Include:

    -   Summary of change and motivation.

    -   Any new configs, defaults, or interfaces.

    -   Usage examples or test snippets if applicable.

    -   Backwards-compatibility notes if relevant.

-   If introducing dependencies or affecting performance budgets, call that out explicitly.

Code and Documentation Style
----------------------------

-   Prefer clear, modular code over cleverness.

-   Document key functions/classes with short docstrings describing inputs, outputs, and side effects.

-   Log key signals where useful (e.g., candidate scores, safety penalties, neuromodulator vector) to support reproducibility and diagnosis.

-   For pseudocode/spec contributions, use concise, unambiguous descriptions and default values where possible.

Community Standards
-------------------

-   Be respectful and constructive.

-   Critique ideas, not people.

-   Prefer proposals with testable claims, measurable metrics, or minimal examples.

Questions and Discussion
------------------------

-   If uncertain about direction, open an issue before large changes.

-   For algorithmic suggestions, include a brief literature pointer or rationale when possible.

How to Submit a PR
------------------

1.  Fork the repository.

2.  Create a feature branch from main.

3.  Commit changes with clear messages.

4.  Open a Pull Request to main.

5.  In the PR:

    -   Explain your change meaningfully.

    -   Describe how you tested it (or how it can be tested).

    -   Note any follow-up work you recommend.

Thanks again for contributing---thoughtful refinements now will significantly accelerate the Isaac Sim and Colab reimplementation and help make the architecture robust, testable, and useful to the broader research community.