---
date: 2026-10-24
slug: fca-ai-failures-near-misses
title: "AI near-misses as incidents: what the UK regulator's latest message means for fraud models"
description: On 19 October 2026 the UK Financial Conduct Authority said incident reporting must extend to significant AI failures and near-misses, and published its good practices for AI in financial services. Why it matters for any team running machine learning in insurance.
authors:
  - abdellatif
comments: true
---

# AI near-misses as incidents: what the UK regulator's latest message means for fraud models

On 19 October 2026, the UK **Financial Conduct Authority (FCA)** sent a clear signal to banks and insurers. Its existing rules on reporting major incidents should also cover **significant AI failures, and even near-misses**. The same day, it published a review of good and poor practice for AI in financial services.

The FCA has no authority in Morocco. Still, its message is worth reading closely for anyone who runs machine-learning models in a regulated business. Most of it applies just as well to a classic fraud-detection model as to the latest large language model.

<!-- more -->

## What was said

The message came in a speech by **Jessica Rusu**, the FCA's chief data, information and intelligence officer, at the Bloomberg UK Policy Series. The FCA has published the speech as drafted. Its main points:

- **Incidents:** firms already have to report major incidents. In her words, that duty "must extend to significant AI failures and near-misses".
- **Agentic finance:** the FCA plans to work with industry, other regulators and AI labs on a framework for AI agents acting in finance. It would cover agent identity, delegated authority, accountability and execution.
- **Concentration risk:** firms should understand their supplier dependencies, including reliance on a single model. They should stress-test their systems, plan for failure and keep fallback options.
- **Continuous assurance:** an agentic system that is safe on day one can behave differently later, for example when the underlying model changes. Assurance has to be ongoing.
- **Disclaimers:** AI providers are asked to state clearly the limits of the financial information their models produce.
- **Adoption:** citing a joint FCA and Bank of England survey due the following week, she said 86% of firms now use AI.

At this stage this is a direction of travel, not a new rule. No consultation on AI incident reporting has been opened, and the FCA had not yet defined what counts as a "significant" failure or a near-miss.

## What the FCA's good-practice review says

The publication, *AI in financial services: good practice and areas for improvement*, covers firms that develop, buy, deploy or use AI. It is organised around three themes: the **Consumer Duty** and vulnerable customers, **governance and accountability**, and **operational resilience**. It creates no new requirements and builds on existing rules. Among the practices it highlights:

- keeping **auditable records** of prompts, outputs, model versions, filters and escalation decisions;
- **monitoring real-world outcomes and model drift** in a forward-looking way, not only technical metrics;
- requiring **advance notice of significant model changes** from vendors in contracts;
- designing systems that can switch providers, with **contingency and exit plans**;
- giving AI risks a **named senior owner**, accountable from the design stage.

The weaknesses it lists mirror these practices: dependence on a single provider without an exit plan, governance done after the fact, and monitoring that watches the model but not the outcomes for customers.

## Why this matters for fraud detection

Fraud models are a textbook case of AI that affects customers. A model that scores claims decides, in practice, which files get a fast payout and which get an investigation. Read with that in mind, the FCA's points translate quite directly. What follows is my own analysis.

**What a near-miss could look like.** Imagine a model update that suddenly doubles the number of legitimate claims flagged as suspicious. Investigators catch it within a week and roll back. No customer was harmed for long, but many waited longer than they should have. Under the approach the FCA describes, that is exactly the kind of event a firm may one day need to detect, document and possibly report.

**What I would put in place, whatever the jurisdiction:**

1. **Version everything.** Training data snapshot, features, model, thresholds and the code that applies them. You cannot explain an incident you cannot reproduce.
2. **Monitor outcomes, not just AUC.** Track alert rates, investigation outcomes, false positives and processing delays, by segment. Drift in the population often shows up there before it shows up in model metrics.
3. **Define incident thresholds in advance.** Decide now which change in alert volume or precision triggers an investigation, a rollback or an escalation.
4. **Keep a fallback.** A previous model version or a rules-based baseline you can switch to while you investigate.
5. **Name an owner.** A model in production needs an accountable person outside the data science team, who understands what it does and signs off on changes.

None of this is new to good MLOps practice. What is changing is that regulators are starting to expect it, and to call the failures by their name.

## Sources

- FCA, [Building the future of finance in the age of AI](https://www.fca.org.uk/news/speeches/building-future-finance-age-ai), speech by Jessica Rusu (19 October 2026)
- FCA, [AI in financial services: good practice and areas for improvement](https://www.fca.org.uk/publications/good-poor-practice/ai-financial-services) (19 October 2026, updated 20 October 2026)
