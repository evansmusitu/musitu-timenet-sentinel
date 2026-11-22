'use strict';
// Run the actual public HTML's inline script with a minimal DOM surface.
// No model inference or network calls: this tests real recorded-case interactions.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('docs/index.html', 'utf8');
const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
assert.equal(scripts.length, 1, 'A single inline script must remain inspectable');
new vm.Script(scripts[0][1], { filename: 'docs/index.html' });
class El {
  constructor(id) { this.id = id; this.value = ''; this.textContent = ''; this.innerHTML = ''; this.children = []; this.listeners = {}; this.attributes = {}; }
  addEventListener(name, fn) { this.listeners[name] = fn; }
  appendChild(el) { this.children.push(el); }
  setAttribute(name, val) { this.attributes[name] = String(val); }
  invoke(name) { assert.ok(this.listeners[name], `Missing ${name} handler on ${this.id}`); this.listeners[name](); }
}
const ids = ['engine','model','scatter','prediction','warning','cycle','actual','error',
  'challenge-case','challenge-model','challenge-result','challenge-cycle',
  'challenge-prediction','challenge-warning','challenge-detail','challenge-inspect','challenge-monitor'];
const els = Object.fromEntries(ids.map(id => [id,new El(id)]));
els.model.value = 'temporal';
els['challenge-model'].value = 'temporal';
els['challenge-case'].value = '41';
const document = {getElementById(id) { assert.ok(els[id], `Unknown DOM target ${id}`); return els[id]; },
  createElement: n => new El(n), createElementNS: (ns,n) => new El(n) };
const sandbox = { document, console };
vm.runInNewContext(scripts[0][1], sandbox, { filename: 'docs/index.html', timeout: 5000 });
assert.equal(els['challenge-warning'].textContent,'WARNING ≤30');
assert.doesNotMatch(els['challenge-result'].textContent,/Official remaining life:/,
  'Actual outcome must be hidden before the human decision');
els['challenge-inspect'].invoke('click');
assert.match(els['challenge-result'].textContent, /Correct call/);
assert.match(els['challenge-result'].textContent, /18 cycles/);
els['challenge-case'].value='91'; els['challenge-case'].invoke('change');
assert.doesNotMatch(els['challenge-result'].textContent,/Official remaining life:/);
els['challenge-inspect'].invoke('click');
assert.match(els['challenge-result'].textContent, /False alarm/);
assert.match(els['challenge-result'].textContent, /38 cycles/);
els['challenge-case'].value='31'; els['challenge-case'].invoke('change');
els['challenge-monitor'].invoke('click');
assert.match(els['challenge-result'].textContent, /Missed warning/);
assert.match(els['challenge-result'].textContent, /8 cycles/);
els['challenge-case'].value='1'; els['challenge-case'].invoke('change');
els['challenge-monitor'].invoke('click');
assert.match(els['challenge-result'].textContent, /Correct call/);
assert.match(els['challenge-result'].textContent, /112 cycles/);
assert.match(html,/These are deliberately selected examples/);
assert.match(html,/not OpenTime 5B/i);
console.log('PASS: operator replay cases, decision reveal, and anti-overclaim copy');
