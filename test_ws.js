import fs from 'fs';
import WebSocket from 'ws';

const authData = JSON.parse(fs.readFileSync('.tmp/auth_info.json', 'utf8'));
console.log('authData:', authData);

const ws = new WebSocket(`ws://localhost:${authData.nlPort}?extensionId=org.consciousnode.brymar.trainer&connectToken=${authData.nlConnectToken}`);
ws.on('open', () => console.log('Connected!'));
ws.on('error', (err) => console.log('Error:', err.message));
