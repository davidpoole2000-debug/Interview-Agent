// server.js (ESM) — HF multi-model interviewer + 3-way assessor comparison + prompt-drafting exercise
import "dotenv/config";

import fs from "fs";
import path from "path";
import express from "express";
import OpenAI from "openai";
import { fileURLToPath } from "url";

/* ==========================
   Paths / App
========================== */

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.json({ limit: "2mb" }));

/* ==========================
   HF Client (Router)
========================== */

const hf = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: process.env.HF_TOKEN,
});

/* ==========================
   Models & Settings (HF-only)
========================== */

const HF_INTERVIEW_MODEL =
  process.env.HF_INTERVIEW_MODEL || "meta-llama/Llama-3.1-8B-Instruct";

const HF_RESPONDENT_MODEL_A =
  process.env.HF_RESPONDENT_MODEL_A || "meta-llama/Llama-3.1-8B-Instruct";

const HF_RESPONDENT_MODEL_B =
  process.env.HF_RESPONDENT_MODEL_B || "Qwen/Qwen2.5-7B-Instruct";

const HF_RESPONDENT_MODEL_C =
  process.env.HF_RESPONDENT_MODEL_C || "google/gemma-2-9b-it";

const HF_ASSESS_MODEL_1 =
  process.env.HF_ASSESS_MODEL_1 || "meta-llama/Llama-3.1-8B-Instruct"; // strict

const HF_ASSESS_MODEL_2 =
  process.env.HF_ASSESS_MODEL_2 || "Qwen/Qwen2.5-7B-Instruct"; // diverse

const HF_ASSESS_MODEL_3 =
  process.env.HF_ASSESS_MODEL_3 || "google/gemma-2-9b-it"; // diverse #2

const MAX_FOLLOWUPS_PER_SKILL = Number(process.env.MAX_FOLLOWUPS_PER_SKILL ?? 1);
const PORT = Number(process.env.PORT ?? 3000);

/* ==========================
   Practical prompt-drafting exercise
========================== */

const EXERCISE_PROMPT = `
Practical Prompting Exercise (write the prompt you would paste into an LLM)

Goal:
You have a long, complex document. Draft a SINGLE prompt that will reliably produce a 2-minute executive brief.

Output requirements:
- EXACTLY 5 bullets
- Each bullet must include: (a) the claim, (b) why it matters, (c) a citation to the document using section headings (not page numbers)
- Include "Key Risks" and "Open Questions" within the 5 bullets (not separate sections)

Grounding & accuracy requirements:
- The model must NOT invent facts. If the doc doesn't support a claim, it must say "Not stated in document."
- Require the model to quote short phrases (<=12 words) from the doc to support each bullet.
- Require the model to provide an "Uncertainties" line under any bullet that is partially supported.

Verification Plan requirements (still inside your prompt):
- Add a final section called "Verification Plan" describing:
  1) what you will check
  2) how you detect hallucinations/confidently-wrong outputs
  3) what you do if output is confidently wrong (rerun steps, tighter constraints, escalation/human review)

Deliverable:
Return ONLY the prompt text you would paste into the LLM.
`.trim();

/* ==========================
   In-memory state
========================== */

const sessions = new Map();

/* ==========================
   Utils
========================== */

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function clampInt(n, min = 1, max = 9) {
  const x = Number.isFinite(n) ? n : parseInt(n, 10);
  if (!Number.isFinite(x)) return min;
  return Math.max(min, Math.min(max, Math.round(x)));
}

function safeFilename(name) {
  return (name || "candidate")
    .trim()
    .replace(/[^\w\-]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function cleanJSON(text) {
  let t = (text || "").trim();
  t = t.replace(/^\s*`{1,3}\s*json\s*/i, "");
  t = t.replace(/^\s*`{1,3}\s*/i, "");
  t = t.replace(/\s*`{1,3}\s*$/i, "");
  const a = t.indexOf("{");
  const b = t.lastIndexOf("}");
  if (a !== -1 && b !== -1 && b > a) t = t.slice(a, b + 1);
  return t.trim();
}

async function callChatJSON({ model, messages, maxAttempts = 3, temperature = 0.2 }) {
  let last = "";
  let delay = 450;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const r = await hf.chat.completions.create({
        model,
        messages,
        temperature,
      });

      last = r?.choices?.[0]?.message?.content || "";
      const cleaned = cleanJSON(last);
      if (!cleaned) throw new Error("Model returned empty content.");

      try {
        return JSON.parse(cleaned);
      } catch {
        // One repair attempt: ask model to output strict JSON
        const fix = await hf.chat.completions.create({
          model,
          messages: [
            { role: "system", content: "Return valid JSON only. No markdown. No backticks." },
            { role: "user", content: "Fix into valid JSON only:\n\n" + last },
          ],
          temperature: 0,
        });

        const fixedText = fix?.choices?.[0]?.message?.content || "";
        const fixedClean = cleanJSON(fixedText);
        if (!fixedClean) throw new Error("JSON-fix step returned empty content.");
        return JSON.parse(fixedClean);
      }
    } catch (e) {
      const msg = String(e?.message || e);
      const is429 = msg.includes("429") || msg.toLowerCase().includes("rate limit");
      const isTransient =
        is429 ||
        msg.toLowerCase().includes("timeout") ||
        msg.toLowerCase().includes("temporarily") ||
        msg.toLowerCase().includes("fetch failed") ||
        msg.toLowerCase().includes("empty content");

      if (attempt === maxAttempts || !isTransient) throw new Error(msg);
      await sleep(delay);
      delay = Math.min(delay * 2, 5000);
    }
  }

  throw new Error("callChatJSON failed unexpectedly.");
}

async function callChatText({ model, messages, temperature = 0.7 }) {
  const r = await hf.chat.completions.create({ model, messages, temperature });
  return (r?.choices?.[0]?.message?.content || "").trim();
}

function loadMatrix() {
  const p = path.join(__dirname, "ai_skills_matrix.json");
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

function getCoverage(transcript, skillNames) {
  const answered = new Set((transcript || []).map((t) => t.skill).filter(Boolean));
  const total = skillNames.length;
  const count = skillNames.filter((s) => answered.has(s)).length;
  return { count, total, fraction: total ? count / total : 0 };
}

function score1to9_to_0to100(score1to9) {
  const s = clampInt(score1to9, 1, 9);
  return Math.round(((s - 1) / 8) * 100);
}

function weightedSkillsScore0to100(skillScores, skills) {
  const wMap = new Map(skills.map((s) => [s.name, s.weight]));
  let totalW = 0;
  let weighted = 0;

  for (const ss of skillScores) {
    const w = wMap.get(ss.skill) ?? 0;
    totalW += w;
    weighted += w * ss.score_1_to_9;
  }

  if (totalW <= 0) return 0;
  const avg = weighted / totalW; // 1..9
  return score1to9_to_0to100(avg);
}

function compareMulti(results) {
  const skills = results[0]?.norm?.skill_scores?.map((s) => s.skill) || [];
  const rows = skills.map((skill) => {
    const scores = results.map((r) => {
      const found = (r.norm.skill_scores || []).find((x) => x.skill === skill);
      return found ? found.score_1_to_9 : 1;
    });
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    const spread = max - min;
    return { skill, scores, min, max, spread };
  });
  rows.sort((a, b) => b.spread - a.spread);
  return { rows, biggest_disagreements: rows.slice(0, 5) };
}

function pickRespondent(which) {
  const w = String(which || "A").toUpperCase();
  if (w === "B") return HF_RESPONDENT_MODEL_B;
  if (w === "C") return HF_RESPONDENT_MODEL_C;
  return HF_RESPONDENT_MODEL_A;
}

/* ==========================
   Interview plan + followups (Interviewer)
========================== */

async function buildInterviewPlan(matrix) {
  const plan = await callChatJSON({
    model: HF_INTERVIEW_MODEL,
    messages: [
      { role: "system", content: "You are a structured interviewer. Return JSON only." },
      {
        role: "user",
        content: `Create an interview plan with ONE strong starter question per skill.
Return JSON ONLY:
{ "questions": [ { "skill": string, "question": string } ] }

Rules:
- concise, evidence-seeking
- ask for concrete examples, tools/platforms, measurable outcomes, validation, failure handling

Skills matrix:
${JSON.stringify(matrix.skills, null, 2)}
`,
      },
    ],
  });

  if (!Array.isArray(plan.questions) || plan.questions.length === 0) {
    throw new Error("Interview plan missing questions.");
  }
  return plan.questions;
}

async function decideFollowup({ skillName, recentQA }) {
  if (!MAX_FOLLOWUPS_PER_SKILL) return { need_followup: false, followup_question: null };

  const decision = await callChatJSON({
    model: HF_INTERVIEW_MODEL,
    messages: [
      { role: "system", content: "Decide if a follow-up is needed. Return JSON only." },
      {
        role: "user",
        content: `Skill: ${skillName}

Recent Q&A:
${JSON.stringify(recentQA, null, 2)}

Return JSON ONLY:
{ "need_followup": boolean, "followup_question": string|null }

Ask a follow-up if evidence is vague or missing:
- concrete example
- tools/platforms
- measurable outcomes
- validation
- failure handling
`,
      },
    ],
  });

  return {
    need_followup: !!decision.need_followup,
    followup_question: decision.followup_question || null,
  };
}

/* ==========================
   Assessment prompt builder
========================== */

function buildAssessmentMessages({ matrix, candidateName, transcript, exerciseAnswer, stance }) {
  const skillNames = matrix.skills.map((s) => s.name);

  const stanceText =
    stance === "strict"
      ? "Be STRICT and evidence-only. Do not infer missing details. If not explicitly evidenced, it does not count."
      : stance === "balanced"
      ? "Be BALANCED. Still evidence-led, but allow reasonable interpretation if the candidate clearly implies details."
      : "Be DIVERSE: produce an independent view and critique. Still avoid inventing evidence.";

  return [
    {
      role: "system",
      content:
        "You are an interview assessor. Return STRICT JSON only. No markdown. No backticks. Do not invent evidence.",
    },
    {
      role: "user",
      content: `Assess candidate "${candidateName}" against the FULL skills matrix.

ASSURANCE:
${stanceText}

SCORING RULE (skills):
- For each skill, choose the SINGLE highest 1–9 level supported by evidence in transcript/exercise.
- If evidence is partial between levels, score the LOWER level.
- If no relevant evidence, score 1 with "Insufficient Evidence".

EXERCISE:
The exercise is a prompt-drafting task. Score it using this rubric (1–9) based ONLY on the exercise answer prompt.

EXERCISE SCORING (1–9) — PROMPT QUALITY RUBRIC:
1–2: Vague prompt, missing format constraints, no grounding, no verification plan.
3–4: Some structure, but weak constraints; minimal grounding; verification plan superficial.
5–6: Clear objective and format; some grounding rules; basic verification plan; limited failure handling detail.
7–8: Strong constraints, explicit grounding/citation/quoting rules, uncertainty handling, step-by-step method, solid verification + escalation.
9: Excellent prompt engineering: tight spec, anti-hallucination controls, explicit refusal rules, traceability (headings + quotes), quality checklist, rerun strategy, escalation path, and clear success criteria.

Return JSON ONLY with EXACT structure:
{
  "recommendation": "Strong Hire|Hire|Lean Hire|Lean No|No|Insufficient Evidence",
  "skill_scores": [
    {
      "skill": string,
      "score_1_to_9": number,
      "confidence_low_med_high": "low|medium|high",
      "assessor_stance": "strict|balanced|diverse",
      "why_this_score": string,
      "what_was_missing_for_next_level": string,
      "evidence_bullets": [string, string, string],
      "risks": [string, string],
      "followups": [string, string]
    }
  ],
  "exercise_score_1_to_9": number,
  "exercise_why_this_score": string,
  "exercise_what_was_missing_for_next_level": string,
  "exercise_evidence_bullets": [string, string, string],
  "summary": [string, string, string, string, string]
}

You MUST include exactly these skills:
${JSON.stringify(skillNames, null, 2)}

Skills matrix:
${JSON.stringify(matrix.skills, null, 2)}

Transcript:
${JSON.stringify(transcript, null, 2)}

Exercise prompt (what candidate was asked):
${EXERCISE_PROMPT}

Exercise answer (candidate's drafted prompt):
${exerciseAnswer}
`,
    },
  ];
}

function normalizeAssessorResult(matrix, raw, stanceLabel) {
  const skillNames = matrix.skills.map((s) => s.name);
  const incoming = Array.isArray(raw.skill_scores) ? raw.skill_scores : [];
  const map = new Map(incoming.map((x) => [String(x.skill || "").trim(), x]));

  const skill_scores = skillNames.map((name) => {
    const x = map.get(name);
    if (!x) {
      return {
        skill: name,
        score_1_to_9: 1,
        confidence_low_med_high: "low",
        assessor_stance: stanceLabel,
        why_this_score: "Insufficient evidence in transcript/exercise.",
        what_was_missing_for_next_level:
          "Provide a concrete example: tools used, steps taken, measurable outcome, validation, and failure handling.",
        evidence_bullets: ["No evidence captured for this skill."],
        risks: ["Evidence gap", "Unverified capability"],
        followups: ["Give a specific example you personally delivered.", "How did you validate and handle failures?"],
      };
    }

    return {
      skill: name,
      score_1_to_9: clampInt(x.score_1_to_9 ?? x.score, 1, 9),
      confidence_low_med_high: x.confidence_low_med_high || "medium",
      assessor_stance: x.assessor_stance || stanceLabel,
      why_this_score: String(x.why_this_score || ""),
      what_was_missing_for_next_level: String(x.what_was_missing_for_next_level || ""),
      evidence_bullets: Array.isArray(x.evidence_bullets) ? x.evidence_bullets.slice(0, 3) : [],
      risks: Array.isArray(x.risks) ? x.risks.slice(0, 2) : [],
      followups: Array.isArray(x.followups) ? x.followups.slice(0, 2) : [],
    };
  });

  return {
    recommendation: String(raw.recommendation || "Lean Hire"),
    skill_scores,
    exercise_score_1_to_9: clampInt(raw.exercise_score_1_to_9, 1, 9),
    exercise_why_this_score: String(raw.exercise_why_this_score || ""),
    exercise_what_was_missing_for_next_level: String(raw.exercise_what_was_missing_for_next_level || ""),
    exercise_evidence_bullets: Array.isArray(raw.exercise_evidence_bullets)
      ? raw.exercise_evidence_bullets.slice(0, 3)
      : [],
    summary: Array.isArray(raw.summary) ? raw.summary.slice(0, 6) : [],
  };
}

function renderReportMarkdown({
  assessorName,
  modelName,
  candidateName,
  role,
  coverage,
  overall0to100,
  recommendation,
  result,
}) {
  const lines = [];
  lines.push(`# AI Capability Assessment Report — ${assessorName}`);
  lines.push(`**Model:** ${modelName}`);
  lines.push(`**Candidate:** ${candidateName}`);
  lines.push(`**Role assessed:** ${role}`);
  lines.push(
    `**Coverage:** ${coverage.skills_answered}/${coverage.skills_total} skills answered (${Math.round(
      coverage.fraction * 100
    )}%)`
  );
  lines.push(`**Overall score:** ${overall0to100}/100`);
  lines.push(`**Recommendation:** ${recommendation}`);
  lines.push("");

  if (Array.isArray(result.summary) && result.summary.length) {
    lines.push("## Executive Summary");
    for (const s of result.summary.slice(0, 6)) lines.push(`- ${s}`);
    lines.push("");
  }

  lines.push("## Skill Scores");
  for (const s of result.skill_scores || []) {
    lines.push(`### ${s.skill}`);
    lines.push(`- **Score:** ${s.score_1_to_9}/9`);
    lines.push(`- **Confidence:** ${s.confidence_low_med_high || "medium"}`);
    lines.push(`- **Assessor stance:** ${s.assessor_stance || "diverse"}`);
    if (s.why_this_score) lines.push(`- **Why this score:** ${s.why_this_score}`);
    if (s.what_was_missing_for_next_level)
      lines.push(`- **Missing for next level:** ${s.what_was_missing_for_next_level}`);
    if (Array.isArray(s.evidence_bullets) && s.evidence_bullets.length) {
      lines.push(`- **Evidence:**`);
      for (const b of s.evidence_bullets.slice(0, 3)) lines.push(`  - ${b}`);
    }
    if (Array.isArray(s.followups) && s.followups.length) {
      lines.push(`- **Suggested follow-ups:**`);
      for (const f of s.followups.slice(0, 2)) lines.push(`  - ${f}`);
    }
    lines.push("");
  }

  lines.push("## Practical Exercise — Prompt Drafting");
  lines.push(`- **Score:** ${clampInt(result.exercise_score_1_to_9, 1, 9)}/9`);
  if (result.exercise_why_this_score) lines.push(`- **Why this score:** ${result.exercise_why_this_score}`);
  if (result.exercise_what_was_missing_for_next_level)
    lines.push(`- **Missing for next level:** ${result.exercise_what_was_missing_for_next_level}`);
  if (Array.isArray(result.exercise_evidence_bullets) && result.exercise_evidence_bullets.length) {
    lines.push(`- **Evidence:**`);
    for (const b of result.exercise_evidence_bullets.slice(0, 3)) lines.push(`  - ${b}`);
  }
  lines.push("");
  return lines.join("\n");
}

/* ==========================
   UI
========================== */

app.get("/", (req, res) => {
  res.type("html").send(`<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>AI Interview Agent</title>
  <style>
    body { font-family: system-ui; max-width: 1200px; margin: 32px auto; padding: 0 12px; }
    input, textarea { width: 100%; padding: 10px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    textarea { height: 130px; }
    button { padding: 10px 14px; font-size: 16px; margin-top: 8px; margin-right: 8px; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 14px; margin: 12px 0; }
    .muted { color: #666; font-size: 13px; }
    .row { display:flex; gap:12px; }
    .col { flex:1; }
    pre { background:#f6f6f6; padding:12px; border-radius:12px; white-space:pre-wrap; overflow:auto; height: 520px; }
    table { width:100%; border-collapse: collapse; margin-top: 10px;}
    th, td { border-bottom: 1px solid #eee; padding: 8px; text-align: left; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
  </style>
</head>
<body>
  <h1>AI Interview Agent (HF Multi-Model)</h1>
  <p class="muted">
    Interviewer: ${HF_INTERVIEW_MODEL}<br/>
    Respondents: A=${HF_RESPONDENT_MODEL_A} • B=${HF_RESPONDENT_MODEL_B} • C=${HF_RESPONDENT_MODEL_C}<br/>
    Assessors: (1) ${HF_ASSESS_MODEL_1} [strict] • (2) ${HF_ASSESS_MODEL_2} [diverse] • (3) ${HF_ASSESS_MODEL_3} [diverse]
  </p>

  <div class="card">
    <label><b>Candidate name</b></label>
    <input id="name" placeholder="Jane Doe" />
    <label class="muted" style="display:block; margin-top:8px;">Consent (type "yes")</label>
    <input id="consent" placeholder="yes" />
    <div>
      <button onclick="startInterview()">Start interview</button>
      <button onclick="resetAll()">Reset</button>
    </div>
    <div id="status" class="muted"></div>
  </div>

  <div id="qa" class="card" style="display:none;">
    <div class="muted" id="skill"></div>
    <h3 id="question"></h3>

    <label><b>Candidate answer</b></label>
    <textarea id="answer"></textarea>

    <div>
      <button onclick="submitAnswer()">Submit answer</button>
      <button onclick="skipAnswer()">Skip</button>
    </div>

    <div style="margin-top:8px;">
      <span class="muted">Auto-answer:</span>
      <button onclick="autoAnswer('A',3)">A L3</button>
      <button onclick="autoAnswer('A',6)">A L6</button>
      <button onclick="autoAnswer('A',9)">A L9</button>
      <button onclick="autoAnswer('B',3)">B L3</button>
      <button onclick="autoAnswer('B',6)">B L6</button>
      <button onclick="autoAnswer('B',9)">B L9</button>
      <button onclick="autoAnswer('C',3)">C L3</button>
      <button onclick="autoAnswer('C',6)">C L6</button>
      <button onclick="autoAnswer('C',9)">C L9</button>
    </div>
  </div>

  <div id="exercise" class="card" style="display:none;">
    <h3>Practical exercise — Draft the prompt</h3>
    <p class="muted">${EXERCISE_PROMPT.replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll("\\n","<br/>")}</p>
    <textarea id="exerciseAnswer" placeholder="Paste ONLY the prompt text you would use..."></textarea>

    <div style="margin-top:8px;">
      <span class="muted">Auto-exercise (draft a prompt):</span>
      <button onclick="autoExercise('A',3)">A L3</button>
      <button onclick="autoExercise('A',6)">A L6</button>
      <button onclick="autoExercise('A',9)">A L9</button>
      <button onclick="autoExercise('B',3)">B L3</button>
      <button onclick="autoExercise('B',6)">B L6</button>
      <button onclick="autoExercise('B',9)">B L9</button>
      <button onclick="autoExercise('C',3)">C L3</button>
      <button onclick="autoExercise('C',6)">C L6</button>
      <button onclick="autoExercise('C',9)">C L9</button>
    </div>

    <div><button onclick="finish()">Assess + Compare</button></div>
  </div>

  <div id="results" class="card" style="display:none;">
    <h3>Assessments (3 assessors)</h3>

    <div class="row" style="margin-top:10px;">
      <div class="col">
        <h4>Assessor 1 (strict)</h4>
        <pre id="report1"></pre>
      </div>
      <div class="col">
        <h4>Assessor 2 (diverse)</h4>
        <pre id="report2"></pre>
      </div>
      <div class="col">
        <h4>Assessor 3 (diverse)</h4>
        <pre id="report3"></pre>
      </div>
    </div>

    <h4 style="margin-top:14px;">Disagreement (spread by skill)</h4>
    <table id="diffTable">
      <thead>
        <tr>
          <th>Skill</th>
          <th>Assessor 1</th>
          <th>Assessor 2</th>
          <th>Assessor 3</th>
          <th>Spread</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>

<script>
  let sessionId = null;

  function setStatus(t){ document.getElementById("status").textContent = t; }

  function showQA(next){
    document.getElementById("qa").style.display = "block";
    document.getElementById("exercise").style.display = "none";
    document.getElementById("results").style.display = "none";
    document.getElementById("skill").textContent = "Skill: " + next.skill;
    document.getElementById("question").textContent = next.question;
    document.getElementById("answer").value = "";
  }

  function showExercise(){
    document.getElementById("qa").style.display = "none";
    document.getElementById("exercise").style.display = "block";
    document.getElementById("results").style.display = "none";
    document.getElementById("exerciseAnswer").value = "";
  }

  async function startInterview(){
    try {
      const name = document.getElementById("name").value || "Candidate";
      const consent = (document.getElementById("consent").value || "").trim().toLowerCase();
      if(consent !== "yes"){ setStatus("Consent required: type 'yes'."); return; }

      setStatus("Starting...");
      const r = await fetch("/api/start", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({ candidateName: name })
      });
      const j = await r.json();
      if(j.error){ setStatus(j.error); return; }

      sessionId = j.sessionId;
      setStatus("Interview started.");
      showQA(j.next);
    } catch(e) {
      setStatus(String(e));
    }
  }

  async function submitAnswer(){
    try {
      if(!sessionId){ setStatus("No session. Click Start interview."); return; }
      const ans = document.getElementById("answer").value || "";

      setStatus("Sending answer...");
      const r = await fetch("/api/answer", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({ sessionId, answer: ans })
      });

      const j = await r.json();
      if(j.error){ setStatus(j.error); return; }

      if(j.done){
        setStatus("Interview questions complete. Please do the exercise.");
        showExercise();
      } else {
        setStatus("Next question ready.");
        showQA(j.next);
      }
    } catch(e) {
      setStatus(String(e));
    }
  }

  async function skipAnswer(){
    document.getElementById("answer").value = "[SKIPPED]";
    await submitAnswer();
  }

  async function autoAnswer(which, level){
    try{
      if(!sessionId){ setStatus("No session."); return; }
      setStatus("Generating auto-answer ("+which+", level "+level+")...");
      const r = await fetch("/api/auto-answer", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({ sessionId, which, level })
      });
      const j = await r.json();
      if(j.error){ setStatus(j.error); return; }
      document.getElementById("answer").value = j.answer || "";
      setStatus("Auto-answer inserted.");
    }catch(e){
      setStatus(String(e));
    }
  }

  async function autoExercise(which, level){
    try{
      if(!sessionId){ setStatus("No session."); return; }
      setStatus("Generating auto-exercise ("+which+", level "+level+")...");
      const r = await fetch("/api/auto-exercise", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({ sessionId, which, level })
      });
      const j = await r.json();
      if(j.error){ setStatus(j.error); return; }
      document.getElementById("exerciseAnswer").value = j.answer || "";
      setStatus("Auto-exercise inserted.");
    }catch(e){
      setStatus(String(e));
    }
  }

  async function finish(){
    try {
      if(!sessionId){ setStatus("No session."); return; }
      const exerciseAnswer = document.getElementById("exerciseAnswer").value || "";

      setStatus("Assessing...");
      const r = await fetch("/api/finish", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({ sessionId, exerciseAnswer })
      });
      const j = await r.json();
      if(j.error){ setStatus(j.error); return; }

      document.getElementById("qa").style.display = "none";
      document.getElementById("exercise").style.display = "none";
      document.getElementById("results").style.display = "block";

      document.getElementById("report1").textContent = j.report1_markdown || "(no report)";
      document.getElementById("report2").textContent = j.report2_markdown || "(no report)";
      document.getElementById("report3").textContent = j.report3_markdown || "(no report)";

      const tbody = document.querySelector("#diffTable tbody");
      tbody.innerHTML = "";
      (j.disagreement?.rows || []).forEach(row => {
        const tr = document.createElement("tr");
        tr.innerHTML =
          "<td>"+row.skill+"</td>"+
          "<td>"+row.scores[0]+"</td>"+
          "<td>"+row.scores[1]+"</td>"+
          "<td>"+row.scores[2]+"</td>"+
          "<td>"+row.spread+"</td>";
        tbody.appendChild(tr);
      });

      setStatus("Done.");
    } catch(e) {
      setStatus(String(e));
    }
  }

  function resetAll(){
    sessionId = null;
    document.getElementById("qa").style.display = "none";
    document.getElementById("exercise").style.display = "none";
    document.getElementById("results").style.display = "none";
    setStatus("Reset.");
  }
</script>
</body>
</html>`);
});

/* ==========================
   API: Start
========================== */

app.post("/api/start", async (req, res) => {
  try {
    const matrix = loadMatrix();
    const candidateName = String(req.body?.candidateName || "Candidate");

    const questions = await buildInterviewPlan(matrix);

    const sessionId = `${Date.now()}_${Math.random().toString(16).slice(2)}`;
    sessions.set(sessionId, {
      candidateName,
      matrix,
      planQuestions: questions,
      idx: 0,
      transcript: [],
      followupsLeft: MAX_FOLLOWUPS_PER_SKILL,
      createdAt: new Date().toISOString(),
    });

    res.json({ sessionId, next: questions[0] });
  } catch (e) {
    res.status(500).json({ error: String(e?.message || e) });
  }
});

/* ==========================
   API: Answer (followups)
========================== */

app.post("/api/answer", async (req, res) => {
  try {
    const sessionId = String(req.body?.sessionId || "");
    const answer = String(req.body?.answer || "");
    const s = sessions.get(sessionId);
    if (!s) return res.status(400).json({ error: "Session not found. Start again." });

    const q = s.planQuestions[s.idx];
    const skillName = String(q.skill || "");
    const question = String(q.question || "");

    s.transcript.push({ skill: skillName, question, answer });

    const recent = s.transcript.filter((t) => t.skill === skillName).slice(-3);

    if (s.followupsLeft > 0 && answer.trim() && answer.trim() !== "[SKIPPED]") {
      const decision = await decideFollowup({ skillName, recentQA: recent });
      if (decision.need_followup && decision.followup_question) {
        s.followupsLeft -= 1;
        return res.json({ done: false, next: { skill: skillName, question: decision.followup_question } });
      }
    }

    s.idx += 1;
    s.followupsLeft = MAX_FOLLOWUPS_PER_SKILL;

    if (s.idx >= s.planQuestions.length) return res.json({ done: true });
    return res.json({ done: false, next: s.planQuestions[s.idx] });
  } catch (e) {
    res.status(500).json({ error: String(e?.message || e) });
  }
});

/* ==========================
   API: Auto-answer (A/B/C)
========================== */

app.post("/api/auto-answer", async (req, res) => {
  try {
    const sessionId = String(req.body?.sessionId || "");
    const which = String(req.body?.which || "A").toUpperCase();
    const level = clampInt(req.body?.level ?? 6, 1, 9);

    const s = sessions.get(sessionId);
    if (!s) return res.status(400).json({ error: "Session not found." });

    const q = s.planQuestions[s.idx];
    if (!q) return res.status(400).json({ error: "No current question." });

    const candidateName = s.candidateName || "Candidate";
    const model = pickRespondent(which);

    const prompt = `
Generate a realistic interview answer (plain text) for:

Candidate: ${candidateName}
Skill: ${q.skill}
Question: ${q.question}

Target capability level (1–9): ${level}

Rules:
- Level 1–3: basic, vague, minimal tools and metrics
- Level 4–6: structured, some tools, some validation, a modest example
- Level 7–9: concrete tools/platforms, measurable outcomes, validation steps, failure handling, monitoring
- Return plain text only.
`.trim();

    const answer = await callChatText({
      model,
      messages: [
        { role: "system", content: "You are simulating a candidate's spoken answer. Be concise but concrete." },
        { role: "user", content: prompt },
      ],
      temperature: 0.8,
    });

    if (!answer) throw new Error("Model returned empty answer.");
    res.json({ answer });
  } catch (e) {
    res.status(500).json({ error: String(e?.message || e) });
  }
});

/* ==========================
   API: Auto-exercise (A/B/C) — generate the drafted PROMPT
========================== */

app.post("/api/auto-exercise", async (req, res) => {
  try {
    const sessionId = String(req.body?.sessionId || "");
    const which = String(req.body?.which || "A").toUpperCase();
    const level = clampInt(req.body?.level ?? 6, 1, 9);

    const s = sessions.get(sessionId);
    if (!s) return res.status(400).json({ error: "Session not found." });

    const candidateName = s.candidateName || "Candidate";
    const model = pickRespondent(which);

    const prompt = `
Generate a realistic candidate response to the exercise below.

Candidate: ${candidateName}
Target capability level (1–9): ${level}

Exercise:
${EXERCISE_PROMPT}

Rules:
- The answer MUST be a single prompt the candidate would paste into an LLM.
- Higher levels should include: explicit role, step-by-step method, strict formatting, grounding rules, quoting rules, uncertainty handling, and a strong verification plan.
- Lower levels should be vague, missing constraints, and weaker on verification/grounding.
- Return plain text only (the prompt text).
`.trim();

    const answer = await callChatText({
      model,
      messages: [
        { role: "system", content: "You are simulating a candidate. Return ONLY the prompt text." },
        { role: "user", content: prompt },
      ],
      temperature: 0.8,
    });

    if (!answer) throw new Error("Model returned empty answer.");
    res.json({ answer });
  } catch (e) {
    res.status(500).json({ error: String(e?.message || e) });
  }
});

/* ==========================
   API: Finish (3 assessors)
========================== */

app.post("/api/finish", async (req, res) => {
  try {
    const sessionId = String(req.body?.sessionId || "");
    const exerciseAnswer = String(req.body?.exerciseAnswer || "");
    const s = sessions.get(sessionId);
    if (!s) return res.status(400).json({ error: "Session not found. Start again." });

    const matrix = s.matrix;
    const candidateName = s.candidateName;
    const role = matrix.role;
    const transcript = s.transcript;

    const skillNames = matrix.skills.map((x) => x.name);
    const cov = getCoverage(transcript, skillNames);

    const coverage = {
      skills_answered: cov.count,
      skills_total: cov.total,
      fraction: Number(cov.fraction.toFixed(2)),
    };

    const raw1 = await callChatJSON({
      model: HF_ASSESS_MODEL_1,
      messages: buildAssessmentMessages({ matrix, candidateName, transcript, exerciseAnswer, stance: "strict" }),
      temperature: 0.2,
    });

    const raw2 = await callChatJSON({
      model: HF_ASSESS_MODEL_2,
      messages: buildAssessmentMessages({ matrix, candidateName, transcript, exerciseAnswer, stance: "diverse" }),
      temperature: 0.25,
    });

    const raw3 = await callChatJSON({
      model: HF_ASSESS_MODEL_3,
      messages: buildAssessmentMessages({ matrix, candidateName, transcript, exerciseAnswer, stance: "diverse" }),
      temperature: 0.25,
    });

    const norm1 = normalizeAssessorResult(matrix, raw1, "strict");
    const norm2 = normalizeAssessorResult(matrix, raw2, "diverse");
    const norm3 = normalizeAssessorResult(matrix, raw3, "diverse");

    function overall(norm) {
      const skills = weightedSkillsScore0to100(norm.skill_scores, matrix.skills);
      return Math.round(0.7 * skills + 0.3 * score1to9_to_0to100(norm.exercise_score_1_to_9));
    }

    const report1 = renderReportMarkdown({
      assessorName: "Assessor 1 (strict)",
      modelName: HF_ASSESS_MODEL_1,
      candidateName,
      role,
      coverage,
      overall0to100: overall(norm1),
      recommendation: norm1.recommendation,
      result: norm1,
    });

    const report2 = renderReportMarkdown({
      assessorName: "Assessor 2 (diverse)",
      modelName: HF_ASSESS_MODEL_2,
      candidateName,
      role,
      coverage,
      overall0to100: overall(norm2),
      recommendation: norm2.recommendation,
      result: norm2,
    });

    const report3 = renderReportMarkdown({
      assessorName: "Assessor 3 (diverse)",
      modelName: HF_ASSESS_MODEL_3,
      candidateName,
      role,
      coverage,
      overall0to100: overall(norm3),
      recommendation: norm3.recommendation,
      result: norm3,
    });

    const disagreement = compareMulti([
      { name: "A1", model: HF_ASSESS_MODEL_1, norm: norm1 },
      { name: "A2", model: HF_ASSESS_MODEL_2, norm: norm2 },
      { name: "A3", model: HF_ASSESS_MODEL_3, norm: norm3 },
    ]);

    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    const outName = `web_assessment_${safeFilename(candidateName)}_${ts}.json`;
    fs.writeFileSync(
      path.join(__dirname, outName),
      JSON.stringify(
        {
          models: {
            interviewer: HF_INTERVIEW_MODEL,
            respondentA: HF_RESPONDENT_MODEL_A,
            respondentB: HF_RESPONDENT_MODEL_B,
            respondentC: HF_RESPONDENT_MODEL_C,
            assessor1: HF_ASSESS_MODEL_1,
            assessor2: HF_ASSESS_MODEL_2,
            assessor3: HF_ASSESS_MODEL_3,
          },
          coverage,
          transcript,
          exercisePrompt: EXERCISE_PROMPT,
          exerciseAnswer,
          report1_markdown: report1,
          report2_markdown: report2,
          report3_markdown: report3,
          disagreement,
        },
        null,
        2
      ),
      "utf8"
    );

    sessions.delete(sessionId);

    res.json({
      report1_markdown: report1,
      report2_markdown: report2,
      report3_markdown: report3,
      disagreement,
    });
  } catch (e) {
    res.status(500).json({ error: String(e?.message || e) });
  }
});

/* ==========================
   Start
========================== */

app.listen(PORT, () => {
  console.log(`✅ Web app running at http://localhost:${PORT}`);
  console.log(`✅ Interviewer: ${HF_INTERVIEW_MODEL}`);
  console.log(`✅ Respondents:`);
  console.log(`   A: ${HF_RESPONDENT_MODEL_A}`);
  console.log(`   B: ${HF_RESPONDENT_MODEL_B}`);
  console.log(`   C: ${HF_RESPONDENT_MODEL_C}`);
  console.log(`✅ Assessors:`);
  console.log(`   1 strict:  ${HF_ASSESS_MODEL_1}`);
  console.log(`   2 diverse: ${HF_ASSESS_MODEL_2}`);
  console.log(`   3 diverse: ${HF_ASSESS_MODEL_3}`);
});