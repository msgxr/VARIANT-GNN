---
name: meta-audit
description: Use when auditing the CAPOS infrastructure itself — are agents, skills, CLAUDE.md, and settings still correctly configured and appropriately sized? Activate when user asks "Claude altyapımızı güçlendir", "sistem hâlâ doğru mu?", "skill/agent güncelle", or when a new recurring workflow pattern is observed that should be systematized. Also activate after major project milestones.
---

# CAPOS Meta-Audit Skill — VARIANT-GNN

When this skill is active, audit the Claude Autonomous Project Operating System (CAPOS) for this project. Evaluate whether the infrastructure is correctly configured, appropriately sized, and serving the project's current needs.

## Audit Scope

### 1. CLAUDE.md Audit
```
Read CLAUDE.md and assess:
[ ] Length appropriate (not too long, not missing key guidance)?
[ ] Competition context section has current, accurate information?
[ ] No section duplicates what's better stated in an agent/skill?
[ ] No section contradicts another section?
[ ] Agent federation reference matches actual agent files?
[ ] Skills reference matches actual skill files?
[ ] Behavioral constitution rules are still all necessary?
[ ] Quality gate table reflects current project needs?
```

### 2. Agent Ecosystem Audit
```
For each agent in .claude/agents/:
[ ] Agent file exists and is readable?
[ ] Agent description matches actual usage pattern?
[ ] Activation criteria are precise enough (not triggering unnecessarily)?
[ ] Activation criteria are broad enough (not missing needed activations)?
[ ] Agent scope is distinct from other agents?
[ ] Agent's known issues section is current?
[ ] Output format is useful and actually being followed?
```

### 3. Skills Ecosystem Audit
```
For each skill in .claude/skills/:
[ ] Skill description triggers at correct times?
[ ] Skill workflow is still accurate (results/context hasn't changed)?
[ ] Skill output format produces actionable output?
[ ] Skill doesn't overlap redundantly with another skill?
[ ] No recurring task type exists without a skill covering it?
```

### 4. Settings Audit
```
Read .claude/settings.local.json:
[ ] Permissions are appropriate (not too broad, not blocking necessary work)?
[ ] Any deprecated permissions to clean up?
[ ] Any missing permissions that would help workflow?
[ ] Hooks configured appropriately?
```

### 5. Context Files Audit
```
For files in .claude/context/ (01_official_source_map, 02_competition_summary,
03_university_category_rules, 04_report_requirements, 05_data_and_metric_rules,
06_team_and_process_rules, 07_uncertainty_log):
[ ] 02_competition_summary.md has current results and architecture?
[ ] 05_data_and_metric_rules.md reflects actual canonical metrics (RESULTS_CANONICAL.json)?
[ ] 06_team_and_process_rules.md identity/process rules still enforced?
[ ] No context file has outdated information that contradicts current reality?
```
Core rules live in .claude/core/PROJECT_RULES.md.

## Improvement Proposal Protocol

For each identified issue, generate a proposal:
```
PROPOSAL [N]: [Short title]
Type: [CLAUDE.md edit / Agent update / New skill / Settings change / Archive]
Issue: [What's wrong or missing]
Proposed Change: [Specific, actionable change]
Priority: [Critical / High / Medium / Low]
Estimated Impact: [How this improves future sessions]
```

## New Skill Detection
```
Signs a new skill should be created:
- User asked for same complex workflow 2+ times manually
- A complex check was repeatedly missed
- A recurring quality gate proved valuable

If detected:
1. Draft new skill description (frontmatter)
2. Draft skill workflow
3. Propose to user
4. Create if approved
```

## Output Format
```
## CAPOS Meta-Audit Report
Date: [today]
Project Phase: [PDR Prep / Finals Prep / Competition]

### Overall Infrastructure Health
[Healthy / Needs Updates / Needs Restructuring]

### CLAUDE.md Status
[Healthy / Issues found]
Issues: [list]
Recommended changes: [specific edits]

### Agent Health
| Agent | Status | Issues | Recommendation |
|---|---|---|---|
| orchestrator | 🟢/🟡/🔴 | | |
| architect | | | |
| scientist | | | |
| debugger | | | |
| verifier | | | |
| documentalist | | | |
| jury-adversary | | | |
| sentinel | | | |
| meta-governor | | | |

### Skill Health (16 skills total)
| Skill | Status | Issues | Recommendation |
|---|---|---|---|
| official-source-guardian | 🟢/🟡/🔴 | | |
| competition-compliance-auditor | | | |
| data-metric-guardian | | | |
| psr-editor | | | |
| pdr-editor | | | |
| report-template-checker | | | |
| experiment-review | | | |
| error-checker | | | |
| variant-gnn-review | | | |
| jury-sim | | | |
| mission-readiness | | | |
| pre-submission-gate | | | |
| reproducibility | | | |
| code-change-verifier | | | |
| git-identity-guardian | | | |
| meta-audit | | | |

### New Skill Proposals
[If any recurring patterns found that lack a skill]

### Settings Recommendations
[Permission additions/removals, hook suggestions]

### Priority Action Plan
1. [Most impactful improvement]
2. ...

### Infrastructure Evolution Assessment
[Is CAPOS serving the project well? What's the next evolution needed?]
```
