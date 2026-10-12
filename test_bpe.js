import { Worker } from 'worker_threads';
const start = Date.now();
const worker = new Worker('./tokenizer/bpe_worker.js');
worker.on('message', (msg) => {
  if (msg.type === 'progress') console.log('progress:', msg.pct);
  if (msg.type === 'done') console.log('done in', Date.now() - start, 'ms');
});
worker.postMessage({ text: 'a'.repeat(80000) + 'b'.repeat(80000), targetVocab: 1000 });
