import fs from 'fs';
import WebSocket from 'ws';

const authData = JSON.parse(fs.readFileSync('.tmp/auth_info.json', 'utf8'));

const ws = new WebSocket(`ws://localhost:${authData.nlPort}?extensionId=org.consciousnode.brymar.trainer&connectToken=\${NL_CONNECTTOKEN}`);
ws.on('open', () => console.log('Connected!'));
ws.on('error', (err) => console.log('Error:', err.message));
