import fs from "fs";
import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config();
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

/* ========= helpers ========= */
function cleanJSON(text) {
  let t = (text || "").trim();
  t = t.replace(/^\s*`{1,3}\s*json\s*/i, "");
  t = t.replace(/^\s*`{1,3}\s*/i, "");
  t = t.replace(/\s*`{1,3}\s*$/i, "");
  const a = t.indexOf("{");
  const b = t.lastIndexOf("}");
  if (a !== -1 && b !== -1) t = t.slice(a, b + 1);
  return t.trim();
}

async function callAIJSON(messages) {
  const r = await client.responses.create({
    model: "gpt-4.1-mini",
    input: messages
  });
  return JSON.parse(cleanJSON(r.output_text));
}

function clamp(n) {
  return Math.max(1, Math.min(9, Math.round(n)));
}

function scoreTo100(s) {
  return Math.round(((s - 1) / 8) * 100);
}

/* ========= exported function ========= */
export async function runAssessment({
  candidateName,
  transcript,
  exerciseAnswer,
  matrix
}) {
  const assessment = await callAIJSON([
    {
      role: "system",
      content:
        "You are an AI interview assessor. Return strict JSON only. No markdown."
    },
    {
      role: "user",
      content: `
Assess candidate "${candidateName}" using the full 9-level skills matrix.

Rules:
- Score each skill 1–9
- Use evidence only from transcript
- Do not infer missing experience
- Use 'Insufficient Evidence' if coverage is weak

Return JSON ONLY with:
recommendation,
skill_scores (skill, score_1_to_9, confidence_low_med_high, evidence_bullets[3], risks[2], followups[2]),
exercise_score_1_to_9,
exercise_evidence_bullets[3],
summary[5]

Skills matrix:
${JSON.stringify(matrix.skills, null, 2)}

Transcript:
${JSON.stringify(transcript, null, 2)}

Exercise:
${exerciseAnswer}
`
    }
  ]);

  const skillScores = assessment.skill_scores.map((s) => ({
    ...s,
    score: clamp(s.score_1_to_9)
  }));

  const skillsAvg =
    skillScores.reduce((a, b) => a + b.score, 0) / skillScores.length;

  const skills100 = scoreTo100(skillsAvg);
  const exercise100 = scoreTo100(clamp(assessment.exercise_score_1_to_9));

  const overall =
    Math.round(skills100 * 0.7 + exercise100 * 0.3);

  return {
    candidate: candidateName,
    recommendation: assessment.recommendation,
    overall_score: overall,
    skill_scores: skillScores,
    exercise: {
      score: clamp(assessment.exercise_score_1_to_9),
      evidence_bullets: assessment.exercise_evidence_bullets
    },
    summary: assessment.summary
  };
}