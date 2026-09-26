const $ = (id) => document.getElementById(id);
const state = { races: [], models: {}, runId: 0 };
const maxHops = 8;

function wikiUrl(title) {
  return `https://en.wikipedia.org/wiki/${encodeURIComponent(title.replaceAll(' ', '_'))}`;
}

function link(title, className = '') {
  const a = document.createElement('a');
  a.href = wikiUrl(title);
  a.target = '_blank';
  a.rel = 'noopener noreferrer';
  a.className = className;
  a.textContent = title;
  return a;
}

function setLaneState(model, label, kind = '') {
  const el = $(`${model}-state`);
  el.textContent = label;
  el.className = `lane-state ${kind}`;
}

function showCourse() {
  const race = state.races[Number($('race-select').value)];
  if (!race) return;
  const course = $('course');
  course.replaceChildren();
  course.append(link(race.start_title, 'course-title'));
  const arrow = document.createElement('span');
  arrow.className = 'course-arrow';
  arrow.textContent = '→';
  course.append(arrow, link(race.target_title, 'course-title'));
}

function showTiming(timing) {
  for (const model of ['base', 'tuned']) {
    for (const metric of ['p50', 'p95']) {
      const value = timing?.[model]?.[`${metric}_ms`];
      $(`${model}-${metric}`).textContent = value == null ? 'n/a' : `${Math.round(value)} ms`;
    }
  }
}

function startTrail(model, race) {
  const trail = $(`${model}-trail`);
  trail.replaceChildren();
  const start = document.createElement('div');
  start.className = 'step';
  const row = document.createElement('div');
  row.className = 'step-top';
  const title = document.createElement('span');
  title.className = 'step-title';
  const num = document.createElement('span');
  num.className = 'step-index';
  num.textContent = 'START';
  title.append(num, link(race.start_title));
  row.append(title);
  start.append(row);
  trail.append(start);
  $(`${model}-hops`).textContent = '0';
}

function addStep(model, step, hop) {
  const el = document.createElement('div');
  el.className = 'step';
  const row = document.createElement('div');
  row.className = 'step-top';
  const title = document.createElement('span');
  title.className = 'step-title';
  const num = document.createElement('span');
  num.className = 'step-index';
  num.textContent = String(hop).padStart(2, '0');
  title.append(num, link(step.choice));
  const time = document.createElement('span');
  time.className = 'step-time';
  time.textContent = `${Math.round(step.model_ms)} ms`;
  row.append(title, time);
  const meta = document.createElement('div');
  meta.className = 'step-meta';
  const bar = document.createElement('span');
  bar.className = 'confidence';
  const fill = document.createElement('span');
  const probability = step.candidates.find((item) => item.index === step.index)?.probability ?? 0;
  fill.style.width = `${Math.min(100, probability * 100)}%`;
  bar.append(fill);
  meta.append(bar, `${(probability * 100).toFixed(1)}% choice probability · ${step.candidates.length} legal links shown`);
  el.append(row, meta);
  $(`${model}-trail`).append(el);
  el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  $(`${model}-hops`).textContent = String(hop);
}

function endTrail(model, message, won = false) {
  const el = document.createElement('div');
  el.className = `end-message ${won ? '' : 'lost'}`;
  el.textContent = message;
  $(`${model}-trail`).append(el);
  setLaneState(model, won ? 'Found it' : 'Stopped', won ? 'won' : 'lost');
}

async function raceModel(model, race, runId) {
  startTrail(model, race);
  if (!state.models[model]?.available) {
    endTrail(model, 'Checkpoint unavailable on this machine.');
    setLaneState(model, 'Unavailable', 'unavailable');
    return;
  }
  setLaneState(model, 'Racing', 'running');
  let current = race.start;
  const visited = [current];
  for (let hop = 1; hop <= maxHops; hop++) {
    if (runId !== state.runId) return;
    try {
      const response = await fetch('/api/step', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, current, target: race.target, visited }),
      });
      const step = await response.json();
      if (runId !== state.runId) return;
      if (!response.ok) throw new Error(step.error || `HTTP ${response.status}`);
      if (step.outcome === 'dead_end') {
        endTrail(model, 'Dead end. No outgoing links.'); return;
      }
      addStep(model, step, hop);
      showTiming(step.timing);
      current = step.index;
      if (current === race.target) {
        endTrail(model, `Destination found in ${hop} ${hop === 1 ? 'hop' : 'hops'}.`, true); return;
      }
      visited.push(current);
    } catch (error) {
      endTrail(model, error.message);
      setLaneState(model, 'Error', 'lost');
      return;
    }
  }
  endTrail(model, `Reached the ${maxHops}-hop limit.`);
}

async function runRace() {
  const race = state.races[Number($('race-select').value)];
  if (!race) return;
  const runId = ++state.runId;
  $('start-button').disabled = true;
  $('setup-note').textContent = `Racing from ${race.start_title} to ${race.target_title}.`;
  await Promise.all(['base', 'tuned'].map((model) => raceModel(model, race, runId)));
  if (runId === state.runId) {
    $('start-button').disabled = false;
    $('setup-note').textContent = 'Race complete. Pick another course or run this one again.';
  }
}

async function load() {
  try {
    const response = await fetch('/api/races');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    state.races = data.races;
    state.models = data.models;
    const select = $('race-select');
    select.replaceChildren();
    data.races.forEach((race, index) => {
      const option = document.createElement('option');
      option.value = String(index);
      option.textContent = `${race.start_title} → ${race.target_title}`;
      select.append(option);
    });
    if (data.races.length) {
      showCourse();
      $('start-button').disabled = false;
      $('setup-note').textContent = `${data.article_count.toLocaleString()} archived articles. Up to ${maxHops} hops per model.`;
    } else {
      const option = document.createElement('option');
      option.textContent = 'No evaluation races yet';
      select.append(option);
      $('setup-note').textContent = 'Add data/wikirace/eval_races.json to enable races.';
    }
    showTiming(data.timing);
    for (const model of ['base', 'tuned']) {
      if (!data.models[model]?.available) setLaneState(model, 'Unavailable', 'unavailable');
    }
  } catch (error) {
    $('setup-note').textContent = `Could not load races: ${error.message}`;
  }
}

$('race-select').addEventListener('change', showCourse);
$('start-button').addEventListener('click', runRace);
load();
